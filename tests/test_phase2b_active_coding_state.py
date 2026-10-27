"""
Comprehensive Test Suite for Phase 2B — Active Coding Problem & Solution State
================================================================================
Verifies that multi-turn coding conversations maintain the SAME coding problem
and SAME generated solution across:
- Flow A: Problem -> Code
- Flow B: Problem -> Code -> Explain
- Flow C: Problem -> Code -> Pseudocode
- Flow D: Problem -> Code -> Complexity
- Flow E: Problem -> Code -> Optimize
- Flow F: Problem -> Code -> Explain -> Pseudocode -> Complexity -> Optimize
- Flow G: Resetting state when a completely new coding problem begins.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from interview_engine import (
    InterviewSession,
    classify_answer_relationship,
    classify_and_understand_utterance,
)


class TestPhase2BActiveCodingState(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()

    # -------------------------------------------------------------------------
    # Flow A: Problem -> Code
    # -------------------------------------------------------------------------
    def test_flow_a_problem_to_code(self):
        """
        Verify that a problem description followed by 'Give me the code.'
        generates the active coding solution and preserves the canonical problem.
        """
        problem_desc = "Given an array of integers nums and an integer target, return indices of the two numbers."
        self.session.active_coding_problem = problem_desc

        rel = classify_answer_relationship("Give me the code.", session=self.session)
        self.assertEqual(rel["strategy_key"], "coding")
        self.assertTrue(rel["is_follow_up"])
        self.assertIn(problem_desc, rel["fused_question"])

        canonical_code = (
            "```python\n"
            "def twoSum(nums: list[int], target: int) -> list[int]:\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        comp = target - num\n"
            "        if comp in seen:\n"
            "            return [seen[comp], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
            "```"
        )
        self.session.record_qa("Give me the code.", canonical_code, category="coding")

        # Must preserve original problem description and save full code block
        self.assertEqual(self.session.active_coding_problem, problem_desc)
        self.assertEqual(self.session.active_coding_solution, canonical_code)
        self.assertEqual(self.session.active_coding_language, "Python")

    # -------------------------------------------------------------------------
    # Flow B: Problem -> Code -> Explain
    # -------------------------------------------------------------------------
    def test_flow_b_problem_to_code_to_explain(self):
        """
        Verify that explaining the approach references the SAME solution
        and does NOT overwrite the canonical code solution.
        """
        problem_desc = "Two Sum: Find two numbers adding up to target."
        self.session.active_coding_problem = problem_desc
        code_block = (
            "```python\n"
            "def twoSum(nums, target):\n"
            "    return [0, 1]\n"
            "```"
        )
        self.session.record_qa("Write the solution.", code_block, category="coding")

        # Turn 2: Explain approach
        rel_explain = classify_answer_relationship("Explain your approach.", session=self.session)
        self.assertEqual(rel_explain["strategy_key"], "code_explanation")
        self.assertTrue(rel_explain["is_follow_up"])
        self.assertIn("Two Sum", rel_explain["fused_question"])

        # Recording explanation must NOT overwrite canonical code
        explain_answer = "I iterate through the array once using a hash map to look up the complement in O(1) time."
        self.session.record_qa("Explain your approach.", explain_answer, category="code_explanation")

        self.assertEqual(self.session.active_coding_solution, code_block)
        self.assertEqual(self.session.active_coding_problem, problem_desc)

    # -------------------------------------------------------------------------
    # Flow C: Problem -> Code -> Pseudocode
    # -------------------------------------------------------------------------
    def test_flow_c_problem_to_code_to_pseudocode(self):
        """
        Verify that requesting pseudocode maintains the SAME solution anchor
        and does NOT overwrite canonical Python code with text/pseudocode blocks.
        """
        problem_desc = "Reverse a Singly Linked List"
        self.session.active_coding_problem = problem_desc
        code_block = (
            "```python\n"
            "def reverseList(head):\n"
            "    prev = None\n"
            "    curr = head\n"
            "    while curr:\n"
            "        nxt = curr.next\n"
            "        curr.next = prev\n"
            "        prev = curr\n"
            "        curr = nxt\n"
            "    return prev\n"
            "```"
        )
        self.session.record_qa("Implement this.", code_block, category="coding")

        # Turn 2: Pseudocode
        rel_pseudo = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel_pseudo["strategy_key"], "pseudocode")
        self.assertTrue(rel_pseudo["is_follow_up"])

        pseudo_answer = (
            "```text\n"
            "prev = None, curr = head\n"
            "while curr is not null:\n"
            "    next_node = curr.next\n"
            "    curr.next = prev\n"
            "    prev = curr\n"
            "    curr = next_node\n"
            "return prev\n"
            "```"
        )
        self.session.record_qa("Give me pseudocode.", pseudo_answer, category="pseudocode")

        # Canonical code must remain Python reverseList, NOT text pseudocode
        self.assertEqual(self.session.active_coding_solution, code_block)

    # -------------------------------------------------------------------------
    # Flow D: Problem -> Code -> Complexity
    # -------------------------------------------------------------------------
    def test_flow_d_problem_to_code_to_complexity(self):
        """
        Verify that asking for time/space complexity anchors to the active solution.
        """
        problem_desc = "Binary Search on Sorted Array"
        self.session.active_coding_problem = problem_desc
        code_block = (
            "```python\n"
            "def search(nums, target):\n"
            "    left, right = 0, len(nums) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if nums[mid] == target: return mid\n"
            "        elif nums[mid] < target: left = mid + 1\n"
            "        else: right = mid - 1\n"
            "    return -1\n"
            "```"
        )
        self.session.record_qa("Code this.", code_block, category="coding")

        rel_comp = classify_answer_relationship("What's the time complexity?", session=self.session)
        self.assertEqual(rel_comp["strategy_key"], "coding_complexity")
        self.assertTrue(rel_comp["is_follow_up"])

        comp_answer = "• **Complexity**: Time: O(log N) | Space: O(1)"
        self.session.record_qa("What's the time complexity?", comp_answer, category="coding_complexity")

        self.assertEqual(self.session.active_coding_solution, code_block)

    # -------------------------------------------------------------------------
    # Flow E: Problem -> Code -> Optimize
    # -------------------------------------------------------------------------
    def test_flow_e_problem_to_code_to_optimize(self):
        """
        Verify that optimization references the initial code solution
        and updates active_coding_solution with the new optimal code block.
        """
        problem_desc = "Longest Substring Without Repeating Characters"
        self.session.active_coding_problem = problem_desc
        initial_code = (
            "```python\n"
            "def lengthOfLongestSubstring(s: str) -> int:\n"
            "    # O(N^2) brute force\n"
            "    pass\n"
            "```"
        )
        self.session.record_qa("Implement this.", initial_code, category="coding")

        rel_opt = classify_answer_relationship("Can we improve the time complexity?", session=self.session)
        self.assertEqual(rel_opt["strategy_key"], "coding_optimization")
        self.assertTrue(rel_opt["is_follow_up"])

        optimized_code = (
            "```python\n"
            "def lengthOfLongestSubstring(s: str) -> int:\n"
            "    char_map = {}\n"
            "    max_len = left = 0\n"
            "    for right, c in enumerate(s):\n"
            "        if c in char_map and char_map[c] >= left:\n"
            "            left = char_map[c] + 1\n"
            "        char_map[c] = right\n"
            "        max_len = max(max_len, right - left + 1)\n"
            "    return max_len\n"
            "```"
        )
        self.session.record_qa("Can we improve the time complexity?", optimized_code, category="coding_optimization")

        # Active solution is updated with the optimized implementation
        self.assertEqual(self.session.active_coding_solution, optimized_code)
        self.assertEqual(self.session.active_coding_problem, problem_desc)

    # -------------------------------------------------------------------------
    # Flow F: Complete 6-Turn Coding Lifecycle
    # Problem -> Code -> Explain -> Pseudocode -> Complexity -> Optimize
    # -------------------------------------------------------------------------
    def test_flow_f_complete_six_turn_coding_lifecycle(self):
        """
        Tests the entire multi-turn pipeline across all 6 steps seamlessly.
        """
        # Step 0: Problem statement identified
        problem_statement = "Valid Parentheses: Determine if the input string is valid."
        self.session.active_coding_problem = problem_statement

        # Step 1: Code
        code_answer = (
            "```python\n"
            "def isValid(s: str) -> bool:\n"
            "    stack = []\n"
            "    mapping = {')': '(', '}': '{', ']': '['}\n"
            "    for char in s:\n"
            "        if char in mapping:\n"
            "            if not stack or stack.pop() != mapping[char]: return False\n"
            "        else: stack.append(char)\n"
            "    return not stack\n"
            "```\n"
            "• **Complexity**: Time: O(N) | Space: O(N)"
        )
        self.session.record_qa("Give me the code.", code_answer, category="coding")
        self.assertIn("def isValid", self.session.active_coding_solution)
        self.assertEqual(self.session.active_coding_problem, problem_statement)

        # Step 2: Explain Approach
        rel_2 = classify_answer_relationship("Walk me through your code.", session=self.session)
        self.assertEqual(rel_2["strategy_key"], "code_explanation")
        self.session.record_qa("Walk me through your code.", "I push open brackets onto a stack...", category="code_explanation")
        self.assertIn("def isValid", self.session.active_coding_solution)

        # Step 3: Pseudocode
        rel_3 = classify_answer_relationship("Give me pseudocode.", session=self.session)
        self.assertEqual(rel_3["strategy_key"], "pseudocode")
        self.session.record_qa("Give me pseudocode.", "```text\nPush to stack\n```", category="pseudocode")
        self.assertIn("def isValid", self.session.active_coding_solution)

        # Step 4: Complexity
        rel_4 = classify_answer_relationship("What is the Big O?", session=self.session)
        self.assertEqual(rel_4["strategy_key"], "coding_complexity")
        self.session.record_qa("What is the Big O?", "Time: O(N) | Space: O(N)", category="coding_complexity")
        self.assertIn("def isValid", self.session.active_coding_solution)

        # Step 5: Optimize
        rel_5 = classify_answer_relationship("Optimize the approach.", session=self.session)
        self.assertEqual(rel_5["strategy_key"], "coding_optimization")
        optimized_code = (
            "```python\n"
            "def isValidOptimized(s: str) -> bool:\n"
            "    # Inlined mapping for maximum lookup speed\n"
            "    stack = []\n"
            "    for c in s:\n"
            "        if c == '(': stack.append(')')\n"
            "        elif c == '{': stack.append('}')\n"
            "        elif c == '[': stack.append(']')\n"
            "        elif not stack or stack.pop() != c: return False\n"
            "    return not stack\n"
            "```"
        )
        self.session.record_qa("Optimize the approach.", optimized_code, category="coding_optimization")
        self.assertIn("isValidOptimized", self.session.active_coding_solution)

    # -------------------------------------------------------------------------
    # Flow G: New Problem Resets State
    # -------------------------------------------------------------------------
    def test_flow_g_new_coding_problem_resets_state(self):
        """
        Verify that when a completely new coding problem starts,
        the previous solution is cleared and the new problem statement is recorded.
        """
        # Problem 1
        self.session.active_coding_problem = "Problem 1: Two Sum"
        self.session.record_qa("Write code.", "```python\ndef twoSum(): pass\n```", category="coding")
        self.assertIn("twoSum", self.session.active_coding_solution)

        # Reset for new problem
        self.session.reset_coding_state()
        self.assertEqual(self.session.active_coding_problem, "")
        self.assertEqual(self.session.active_coding_solution, "")

        # Problem 2 starts
        new_problem = "Problem 2: Invert Binary Tree"
        self.session.active_coding_problem = new_problem
        self.session.record_qa("Implement this.", "```python\ndef invertTree(root): pass\n```", category="coding")

        self.assertEqual(self.session.active_coding_problem, new_problem)
        self.assertIn("invertTree", self.session.active_coding_solution)
        self.assertNotIn("twoSum", self.session.active_coding_solution)


if __name__ == "__main__":
    unittest.main(verbosity=2)
