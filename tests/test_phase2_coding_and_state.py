import unittest
from interview_engine import (
    InterviewSession,
    classify_answer_relationship,
    classify_and_understand_utterance,
    detect_speaker_turn
)
import classifier
import screen_analyzer


class TestPhase2CodingAndState(unittest.TestCase):
    def setUp(self):
        self.session = InterviewSession()

    def test_01_solve_screen_problem_triggers_screen_reference(self):
        """Verify that 'Solve the problem on the screen.' triggers visual reference detection."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

        self.assertTrue(screen_analyzer.is_visual_reference_question("Solve the problem on the screen."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Solve the problem shown on screen."))
        self.assertTrue(screen_analyzer.is_visual_reference_question("Implement the problem shown."))

    def test_02_give_me_the_code_intent(self):
        """Verify 'Give me the code.' is classified as coding rather than generic technical."""
        q_type = classifier.classify_question("Give me the code.")
        self.assertEqual(q_type, "coding")
        rel = classify_answer_relationship("Give me the code.", session=self.session)
        self.assertEqual(rel["strategy_key"], "coding")

    def test_03_implement_this_intent(self):
        """Verify 'Implement this.' is classified as coding."""
        q_type = classifier.classify_question("Implement this.")
        self.assertEqual(q_type, "coding")
        rel = classify_answer_relationship("Implement this.", session=self.session)
        self.assertEqual(rel["strategy_key"], "coding")

    def test_04_now_code_it_and_write_the_solution(self):
        """Verify 'Now code it.' and 'Write the solution.' are classified as coding."""
        self.assertEqual(classifier.classify_question("Now code it."), "coding")
        self.assertEqual(classifier.classify_question("Write the solution."), "coding")

    def test_05_explain_your_solution_and_approach(self):
        """Verify 'Explain your approach' and 'Explain your solution' map to code_explanation."""
        self.assertEqual(classifier.classify_question("Explain your approach."), "code_explanation")
        self.assertEqual(classifier.classify_question("Explain your solution."), "code_explanation")
        self.assertEqual(classifier.classify_question("Walk me through your code."), "code_explanation")

    def test_06_give_me_pseudocode_intent(self):
        """Verify 'Give me pseudocode.' maps to pseudocode intent."""
        self.assertEqual(classifier.classify_question("Give me pseudocode."), "pseudocode")
        self.assertEqual(classifier.classify_question("Write pseudocode for this."), "pseudocode")
        rel = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel["strategy_key"], "pseudocode")

    def test_07_whats_the_complexity_intent(self):
        """Verify 'What's the complexity?' maps to coding_complexity."""
        self.assertEqual(classifier.classify_question("What's the time complexity?"), "coding_complexity")
        self.assertEqual(classifier.classify_question("What is the Big O?"), "coding_complexity")

    def test_08_optimize_it_intent(self):
        """Verify 'Optimize your solution.' and 'Can you optimize it?' map to coding_optimization."""
        self.assertEqual(classifier.classify_question("Optimize your solution."), "coding_optimization")
        self.assertEqual(classifier.classify_question("Can you optimize it?"), "coding_optimization")
        self.assertEqual(classifier.classify_question("Make it faster."), "coding_optimization")

    def test_09_multi_turn_code_explain_pseudocode_complexity_optimize_continuity(self):
        """
        Verify the complete multi-turn flow:
        Turn 1: Code -> Generates canonical code
        Turn 2: Explain Approach -> Anchors to Turn 1 code without overwriting canonical code
        Turn 3: Pseudocode -> Anchors to Turn 1 code without overwriting canonical code
        Turn 4: Complexity -> Anchors to Turn 1 code
        Turn 5: Optimize -> Uses Turn 1 code as anchor and produces optimized code
        """
        # Turn 1: Initial coding solution
        initial_code_answer = (
            "Here is the optimal O(N) Two Sum solution in Python:\n"
            "```python\n"
            "def twoSum(nums: list[int], target: int) -> list[int]:\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
            "```\n"
            "• **Complexity**: Time: O(N) | Space: O(N)"
        )
        self.session.record_qa("Write a function to solve Two Sum in Python.", initial_code_answer, category="coding")
        
        # Verify initial canonical solution is saved
        self.assertIn("def twoSum", self.session.active_coding_solution)
        self.assertEqual(self.session.active_coding_language, "Python")
        self.assertEqual(self.session.active_coding_problem, "Write a function to solve Two Sum in Python.")

        # Turn 2: "Now explain your approach."
        rel_explain = classify_answer_relationship("Now explain your approach.", session=self.session)
        self.assertEqual(rel_explain["strategy_key"], "code_explanation")
        self.assertTrue(rel_explain["is_follow_up"])
        
        # Record explanation answer (NO code block or text-only block)
        explanation_answer = "I iterate through the array once using a hash map to look up the complement in O(1) time."
        self.session.record_qa("Now explain your approach.", explanation_answer, category="code_explanation")
        
        # Verify canonical solution was NOT overwritten by explanation
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # Turn 3: "Give me pseudocode."
        rel_pseudo = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel_pseudo["strategy_key"], "pseudocode")
        self.assertTrue(rel_pseudo["is_follow_up"])
        
        pseudo_answer = (
            "```text\n"
            "Initialize hash_map\n"
            "For each (index, num) in nums:\n"
            "    complement = target - num\n"
            "    if complement in hash_map: return [hash_map[complement], index]\n"
            "    hash_map[num] = index\n"
            "```"
        )
        self.session.record_qa("Give me pseudocode.", pseudo_answer, category="pseudocode")
        
        # Verify canonical solution was NOT overwritten by pseudocode text block
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # Turn 4: "What's the time complexity?"
        rel_comp = classify_answer_relationship("What's the time complexity?", session=self.session)
        self.assertEqual(rel_comp["strategy_key"], "coding_complexity")
        self.assertTrue(rel_comp["is_follow_up"])
        
        comp_answer = "• **Complexity**: Time: O(N) since we visit each element at most once | Space: O(N) for the hash map."
        self.session.record_qa("What's the time complexity?", comp_answer, category="coding_complexity")
        
        # Verify canonical solution is STILL preserved
        self.assertIn("def twoSum", self.session.active_coding_solution)

        # Turn 5: "Can you optimize it?"
        rel_opt = classify_answer_relationship("Can you optimize it?", session=self.session)
        self.assertEqual(rel_opt["strategy_key"], "coding_optimization")
        self.assertTrue(rel_opt["is_follow_up"])
        
        # Optimized code response updates canonical solution
        optimized_code_answer = (
            "```python\n"
            "def twoSumOptimized(nums: list[int], target: int) -> list[int]:\n"
            "    # Two-pointer optimization\n"
            "    pass\n"
            "```"
        )
        self.session.record_qa("Can you optimize it?", optimized_code_answer, category="coding_optimization")
        self.assertIn("twoSumOptimized", self.session.active_coding_solution)

    def test_10_unrelated_explain_statement_must_not_become_coding(self):
        """Unrelated explain questions (e.g. 'Explain Django middleware') must NOT be code_explanation."""
        self.session.active_topic = "Django"
        self.session.active_coding_solution = ""
        self.session.active_coding_problem = ""
        
        q_type = classifier.classify_question("Explain Django middleware architecture.")
        self.assertNotEqual(q_type, "code_explanation")
        self.assertNotEqual(q_type, "coding")

    def test_11_unrelated_optimize_statement_must_not_become_coding(self):
        """Unrelated optimize questions (e.g. 'How do you optimize team velocity?') must NOT be coding_optimization."""
        self.session.active_coding_solution = ""
        self.session.active_coding_problem = ""
        self.session.active_topic = "Leadership"
        
        rel = classify_answer_relationship("How do you optimize team velocity in sprints?", session=self.session)
        self.assertNotEqual(rel["strategy_key"], "coding_optimization")

    def test_12_screen_problem_plus_spoken_reference(self):
        """Verify screen context is merged with spoken reference 'Can you solve this?'"""
        mock_screen = {
            "is_coding_related": True,
            "language": "Python",
            "problem_statement": "Merge two sorted linked lists into one.",
            "code": "def mergeTwoLists(l1, l2): pass",
            "question_type": "coding"
        }
        fused = screen_analyzer.fuse_voice_and_screen_question("Can you solve this?", screen_context=mock_screen)
        self.assertTrue(fused["is_coding"])
        self.assertEqual(fused["question_type"], "coding")
        self.assertIn("Merge two sorted linked lists", fused["refined_question"])


if __name__ == "__main__":
    unittest.main()
