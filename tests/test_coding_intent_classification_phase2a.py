"""
Comprehensive Test Suite for Phase 2A — Coding Intent Classification
=====================================================================
Tests all 30 positive coding command utterances across 5 categories
and 7 negative non-coding control utterances to verify zero false positives.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
import classifier


class TestCodingIntentClassificationPhase2A(unittest.TestCase):

    # =========================================================================
    # 1. DIRECT CODING INTENTS (Expected: 'coding')
    # =========================================================================
    def test_01_direct_coding_intents(self):
        coding_cases = [
            "Give me the code.",
            "Give me code.",
            "Show me the code.",
            "Implement this.",
            "Implement that.",
            "Implement it.",
            "Now code it.",
            "Code this.",
            "Write the solution.",
            "Code the solution."
        ]
        for query in coding_cases:
            with self.subTest(query=query):
                self.assertEqual(
                    classifier.classify_question(query),
                    "coding",
                    f"Failed for query: '{query}'"
                )

    # =========================================================================
    # 2. PSEUDOCODE INTENTS (Expected: 'pseudocode')
    # =========================================================================
    def test_02_pseudocode_intents(self):
        pseudocode_cases = [
            "Give me pseudocode.",
            "Write pseudocode.",
            "Show me pseudocode.",
            "Give me pseudocode for this.",
            "Give me pseudo code."
        ]
        for query in pseudocode_cases:
            with self.subTest(query=query):
                self.assertEqual(
                    classifier.classify_question(query),
                    "pseudocode",
                    f"Failed for query: '{query}'"
                )

    # =========================================================================
    # 3. CODE EXPLANATION INTENTS (Expected: 'code_explanation')
    # =========================================================================
    def test_03_code_explanation_intents(self):
        explanation_cases = [
            "Explain your approach.",
            "Explain the approach.",
            "Explain your code.",
            "Walk me through your code.",
            "Explain the solution.",
            "How does your code work?"
        ]
        for query in explanation_cases:
            with self.subTest(query=query):
                self.assertEqual(
                    classifier.classify_question(query),
                    "code_explanation",
                    f"Failed for query: '{query}'"
                )

    # =========================================================================
    # 4. COMPLEXITY INQUIRIES (Expected: 'coding_complexity')
    # =========================================================================
    def test_04_coding_complexity_intents(self):
        complexity_cases = [
            "What's the time complexity?",
            "What is the Big O?",
            "What's the space complexity?"
        ]
        for query in complexity_cases:
            with self.subTest(query=query):
                self.assertEqual(
                    classifier.classify_question(query),
                    "coding_complexity",
                    f"Failed for query: '{query}'"
                )

    # =========================================================================
    # 5. CODING OPTIMIZATION INTENTS (Expected: 'coding_optimization')
    # =========================================================================
    def test_05_coding_optimization_intents(self):
        optimization_cases = [
            "Optimize your solution.",
            "Optimize this.",
            "Optimize that.",
            "Make this faster.",
            "Optimize the approach.",
            "Can we improve the time complexity?"
        ]
        for query in optimization_cases:
            with self.subTest(query=query):
                self.assertEqual(
                    classifier.classify_question(query),
                    "coding_optimization",
                    f"Failed for query: '{query}'"
                )

    # =========================================================================
    # 6. NEGATIVE NON-CODING CONTROLS (Zero false positive verification)
    # =========================================================================
    def test_06_non_coding_negative_controls(self):
        coding_intents = {"coding", "pseudocode", "code_explanation", "coding_complexity", "coding_optimization"}
        
        negative_cases = [
            ("Tell me about your project.", "project"),
            ("How does your application work?", "technical"),
            ("Why did you choose Python?", "technical"),
            ("Explain your architecture.", "system_design"),
            ("The implementation was difficult.", "technical"),
            ("The code is already working.", "technical"),
            ("We need to optimize the application.", "technical")
        ]
        for query, expected_category in negative_cases:
            with self.subTest(query=query):
                res = classifier.classify_question(query)
                self.assertNotIn(
                    res,
                    coding_intents,
                    f"Query '{query}' was incorrectly classified as coding intent '{res}'"
                )
                self.assertEqual(res, expected_category)


if __name__ == "__main__":
    unittest.main(verbosity=2)
