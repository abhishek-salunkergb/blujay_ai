"""
Phase 10 Acceptance Test Suite: Conversational Lead-In Acknowledgements
========================================================================
Validates the prefix normalization and downstream preservation of:
A. Standalone acknowledgements
B. Acknowledgement + acknowledgement
C. Acknowledgement + social/conversational statement
D. Acknowledgement + technical question
E. Acknowledgement + coding command
F. Follow-up continuity & topic preservation
G. Incomplete fragment holding & merging
H. Duplicate question suppression
I. Negative adversarial cases
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interview_engine import (
    InterviewSession,
    classify_and_understand_utterance,
    classify_answer_relationship
)


class TestPhase10LeadInAcknowledgements(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()

    # --------------------------------------------------------------------------
    # CATEGORY A: Standalone Acknowledgements (MUST NOT GENERATE)
    # --------------------------------------------------------------------------
    def test_category_a_standalone_acks(self):
        standalone_queries = ["Okay.", "Yeah.", "Right.", "Sure.", "Got it."]
        for query in standalone_queries:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Standalone acknowledgement '{query}' must NOT generate an answer"
                )
                self.assertEqual(turn["utterance_type"], "ACKNOWLEDGEMENT")

    # --------------------------------------------------------------------------
    # CATEGORY B: Acknowledgement + Acknowledgement (MUST NOT GENERATE)
    # --------------------------------------------------------------------------
    def test_category_b_compound_acks(self):
        compound_acks = ["Okay, sounds good.", "Yeah, that's fine.", "Sure, no problem."]
        for query in compound_acks:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Compound acknowledgement '{query}' must NOT generate an answer"
                )
                self.assertEqual(turn["utterance_type"], "ACKNOWLEDGEMENT")

    # --------------------------------------------------------------------------
    # CATEGORY C: Acknowledgement + Social / Conversational Statement (MUST NOT GENERATE)
    # --------------------------------------------------------------------------
    def test_category_c_social_and_statements(self):
        social_queries = [
            "Right, let me rephrase that.",
            "Okay, how are you doing today?"
        ]
        for query in social_queries:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Social/Statement '{query}' must NOT generate an answer"
                )
                self.assertIn(turn["utterance_type"], ["STATEMENT", "GREETING"])

    # --------------------------------------------------------------------------
    # CATEGORY D: Acknowledgement + Technical Question (MUST GENERATE)
    # --------------------------------------------------------------------------
    def test_category_d_technical_questions(self):
        technical_queries = [
            ("Okay, so tell me how you designed the API.", "how you designed the api"),
            ("Yeah that's fine, but why did you choose Redis?", "why did you choose redis"),
            ("Okay, and how do you test your code?", "how do you test your code"),
            ("Right, so what happens if the database goes down?", "what happens if the database goes down")
        ]
        for query, expected_core in technical_queries:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertTrue(
                    turn["should_generate_answer"],
                    f"Technical lead-in '{query}' MUST generate an answer"
                )
                self.assertIn(expected_core, turn["refined_question"].lower())

    # --------------------------------------------------------------------------
    # CATEGORY E: Acknowledgement + Coding Command (MUST GENERATE)
    # --------------------------------------------------------------------------
    def test_category_e_coding_commands(self):
        coding_queries = [
            "Sure, now write the code.",
            "Okay, implement it.",
            "Yeah, now optimize the solution."
        ]
        for query in coding_queries:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertTrue(
                    turn["should_generate_answer"],
                    f"Coding lead-in '{query}' MUST generate an answer"
                )
                self.assertIn(turn["utterance_type"], ["CODING", "CODING_OPTIMIZATION"])

    # --------------------------------------------------------------------------
    # CATEGORY F: Follow-up Continuity & Topic Preservation
    # --------------------------------------------------------------------------
    def test_category_f_followup_continuity(self):
        # Step 1: Base Question
        turn1 = classify_and_understand_utterance("Explain how you designed the API.", session=self.session, source="system")
        self.session.record_qa(turn1["refined_question"], "I built a REST API with FastAPI and Redis caching.", topic="API Design", category="technical")
        
        # Step 2: Follow-up with acknowledgement lead-in
        turn2 = classify_and_understand_utterance("Yeah, but why did you choose Redis?", session=self.session, source="system")
        rel2 = classify_answer_relationship("Yeah, but why did you choose Redis?", session=self.session)
        self.assertTrue(turn2["should_generate_answer"])
        self.assertEqual(rel2.get("relationship_type"), "WHY")
        self.session.record_qa(turn2["refined_question"], "Redis provides in-memory sub-millisecond lookups.", topic="Redis Caching", category="follow_up")

        # Step 3: Follow-up with another lead-in
        turn3 = classify_and_understand_utterance("Okay, and how would you scale it?", session=self.session, source="system")
        rel3 = classify_answer_relationship("Okay, and how would you scale it?", session=self.session)
        self.assertTrue(turn3["should_generate_answer"])
        self.assertEqual(rel3.get("relationship_type"), "HOW")

    # --------------------------------------------------------------------------
    # CATEGORY G: Incomplete Fragments (MUST HOLD, NOT GENERATE)
    # --------------------------------------------------------------------------
    def test_category_g_incomplete_fragments(self):
        fragment_pairs = [
            ("Yeah, what if the tree...", "is empty?", "what if the tree is empty"),
            ("Right, how would you...", "handle authentication?", "how would you handle authentication")
        ]
        for chunk1, chunk2, expected_merged in fragment_pairs:
            with self.subTest(chunk1=chunk1):
                self.session.pending_medium_question = None
                t1 = classify_and_understand_utterance(chunk1, session=self.session, source="system")
                self.assertFalse(t1["should_generate_answer"], f"Fragment '{chunk1}' must hold")
                self.assertEqual(t1["question_state"], "MEDIUM")

                t2 = classify_and_understand_utterance(chunk2, session=self.session, source="system")
                self.assertTrue(t2["should_generate_answer"], f"Continuation '{chunk2}' must complete and generate")
                self.assertIn(expected_merged, t2["refined_question"].lower())

    # --------------------------------------------------------------------------
    # CATEGORY H: Duplicate Suppression
    # --------------------------------------------------------------------------
    def test_category_h_duplicate_suppression(self):
        q = "Okay, so tell me how you designed the API."
        t1 = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertTrue(t1["should_generate_answer"])
        self.session.record_qa(t1["refined_question"], "Answer for API design", category="technical")

        # Repeat same question with same lead-in
        t2 = classify_and_understand_utterance(q, session=self.session, source="system")
        self.assertFalse(t2["should_generate_answer"], f"Duplicate question '{q}' must be suppressed")
        self.assertEqual(t2["utterance_type"], "DUPLICATE")

    # --------------------------------------------------------------------------
    # CATEGORY I: Negative Adversarial Cases (MUST NOT GENERATE)
    # --------------------------------------------------------------------------
    def test_category_i_negative_adversarial(self):
        adversarial_cases = [
            "Okay, I understand.",
            "Yeah, that's correct.",
            "Right, exactly.",
            "Sure, no problem.",
            "Okay, let me check.",
            "Yeah, give me a second."
        ]
        for query in adversarial_cases:
            with self.subTest(query=query):
                turn = classify_and_understand_utterance(query, session=self.session, source="system")
                self.assertFalse(
                    turn["should_generate_answer"],
                    f"Adversarial negative '{query}' must NOT generate an answer"
                )


if __name__ == "__main__":
    unittest.main()
