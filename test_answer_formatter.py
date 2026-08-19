"""
Comprehensive Unit & Integration Test Suite for Parakit Answer Formatter
========================================================================
Tests all formatting principles, question-type structures, code block parsing,
and widget rendering performance.
"""

import sys
import os
import unittest
import time
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import answer_formatter
import classifier


class TestAnswerFormatter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
            cls.text_widget = ScrolledText(cls.root)
            cls.text_widget.tag_config("section_header", foreground="#38bdf8")
            cls.text_widget.tag_config("key_header", foreground="#fbbf24")
            cls.text_widget.tag_config("code_header", foreground="#a78bfa")
            cls.text_widget.tag_config("code_body", foreground="#67e8f9")
            cls.text_widget.tag_config("key_body", foreground="#fef08a")
            cls.text_widget.tag_config("bullet_body", foreground="#f1f5f9")
            cls.text_widget.tag_config("body", foreground="#f8fafc")
        except Exception:
            cls.root = None
            cls.text_widget = None

    @classmethod
    def tearDownClass(cls):
        if cls.root:
            try:
                cls.root.destroy()
            except Exception:
                pass

    # ============================================================
    # TEST 1: TECHNICAL QUESTION FORMAT
    # ============================================================
    def test_01_technical_question_format(self):
        raw = """ANSWER
I implemented authentication using JWT with Django REST Framework.

APPROACH
• User logs in with credentials
• Server validates user and signs JWT token
• Token is used for authenticated API requests

WHY
• Stateless authentication
• Works seamlessly with React frontend

KEY POINT
JWT kept authentication simple and scalable for our microservices."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="technical")
        sections = parsed["sections"]
        titles = [s["title"] for s in sections]

        self.assertIn("ANSWER", titles)
        self.assertIn("APPROACH", titles)
        self.assertIn("WHY", titles)
        self.assertIn("KEY POINT", titles)
        self.assertEqual(sections[0]["content"], "I implemented authentication using JWT with Django REST Framework.")

    # ============================================================
    # TEST 2: SHORT DEFINITION FORMAT
    # ============================================================
    def test_02_short_question_format(self):
        raw = """ANSWER
A REST API is an HTTP-based interface allowing clients and servers to communicate using standard HTTP methods.

KEY:
Stateless + resource-based + standard HTTP methods."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="general")
        sections = parsed["sections"]
        titles = [s["title"] for s in sections]

        self.assertIn("ANSWER", titles)
        self.assertIn("KEY", titles)
        self.assertIn("REST API", parsed["summary"])

    # ============================================================
    # TEST 3: PROJECT QUESTION FORMAT
    # ============================================================
    def test_03_project_question_format(self):
        raw = """ANSWER
• Project: Farmer Marketplace
• Stack: React, Django, PostgreSQL
• Purpose: Direct marketplace for agricultural produce
• Key Feature: Real-time negotiation & orders
• My Role: Full-stack architecture

HOW IT WORKED
1. Farmer lists produce
2. Buyers browse & negotiate
3. Order is confirmed with escrow payment

RESULT
Platform facilitated over 10,000 monthly transactions."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="project")
        titles = [s["title"] for s in parsed["sections"]]

        self.assertIn("ANSWER", titles)
        self.assertIn("HOW IT WORKED", titles)
        self.assertIn("RESULT", titles)

    # ============================================================
    # TEST 4: BEHAVIORAL (STAR) FORMAT
    # ============================================================
    def test_04_behavioral_star_format(self):
        raw = """SITUATION
During Black Friday, our payment processing service experienced high latency under 10x traffic spikes.

TASK
I was responsible for isolating the bottleneck and stabilizing payment checkout without downtime.

ACTION
• Analyzed distributed traces and identified database connection pool exhaustion
• Implemented Redis write-through caching and connection pool scaling

RESULT
Checkout latency dropped by 65% with zero dropped transactions."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="behavioral")
        titles = [s["title"] for s in parsed["sections"]]

        self.assertIn("SITUATION", titles)
        self.assertIn("TASK", titles)
        self.assertIn("ACTION", titles)
        self.assertIn("RESULT", titles)

    # ============================================================
    # TEST 5: CODING QUESTION FORMAT & CODE EXTRACTION
    # ============================================================
    def test_05_coding_question_format(self):
        raw = """APPROACH
Use a hash map to store previously seen elements and check for target complement in O(1) time.

CODE
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []
```

COMPLEXITY
Time: O(n)
Space: O(n)

EDGE CASES
• Empty array
• No solution exists
• Duplicate values summing to target"""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="coding")
        self.assertEqual(len(parsed["code_blocks"]), 1)
        self.assertEqual(parsed["code_blocks"][0]["language"], "python")
        self.assertIn("def two_sum", parsed["code_blocks"][0]["code"])
        self.assertEqual(parsed["complexity"]["time"], "O(n)")
        self.assertEqual(parsed["complexity"]["space"], "O(n)")

    # ============================================================
    # TEST 6: DEBUGGING QUESTION FORMAT
    # ============================================================
    def test_06_debugging_question_format(self):
        raw = """PROBLEM
The IndexError occurs because the loop boundary exceeds the list length when accessing `nums[i + 1]`.

CAUSE
The range was `range(len(nums))` instead of `range(len(nums) - 1)`.

FIX
```python
for i in range(len(nums) - 1):
    if nums[i] == nums[i + 1]:
        return True
```

WHY
Restricting the upper bound prevents accessing out-of-bounds indices on the final iteration."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="debugging")
        titles = [s["title"] for s in parsed["sections"]]

        self.assertIn("PROBLEM", titles)
        self.assertIn("CAUSE", titles)
        self.assertTrue("FIX" in titles or "CODE" in titles)
        self.assertIn("WHY", titles)

    # ============================================================
    # TEST 7: SQL QUESTION FORMAT
    # ============================================================
    def test_07_sql_question_format(self):
        raw = """SQL
```sql
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM employees
) ranked
WHERE rank = 2 LIMIT 1;
```

EXPLANATION
• Uses DENSE_RANK() to correctly handle duplicate top salaries
• Filters for rank = 2 to retrieve the second highest salary

PERFORMANCE
Recommend a B-Tree index on the `salary` column."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="sql")
        self.assertEqual(len(parsed["code_blocks"]), 1)
        self.assertEqual(parsed["code_blocks"][0]["language"], "sql")
        self.assertIn("DENSE_RANK()", parsed["code_blocks"][0]["code"])

    # ============================================================
    # TEST 8: SYSTEM DESIGN QUESTION FORMAT
    # ============================================================
    def test_08_system_design_question_format(self):
        raw = """REQUIREMENTS
• Functional: Shorten URL, redirect to original URL, analytics
• Non-Functional: High availability, low read latency <10ms

ARCHITECTURE
Client -> CDN -> Load Balancer -> API Service -> Redis Cache -> MongoDB / Sharded SQL

COMPONENTS
• Shortening Service: Base62 encoding / Snowflake ID generator
• Redirection Service: High-throughput cached lookup service

SCALABILITY
• Redis cluster for top 20% hot URLs
• Horizontal scaling of stateless redirection workers

TRADE-OFFS
• Eventual consistency for click analytics vs strict consistency for URL creation"""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="system_design")
        titles = [s["title"] for s in parsed["sections"]]

        self.assertIn("REQUIREMENTS", titles)
        self.assertIn("ARCHITECTURE", titles)
        self.assertIn("COMPONENTS", titles)
        self.assertIn("SCALABILITY", titles)
        self.assertIn("TRADE-OFFS", titles)

    # ============================================================
    # TEST 9: FOLLOW-UP QUESTION FORMAT
    # ============================================================
    def test_09_follow_up_question_format(self):
        raw = """SCALABILITY
• Implemented horizontal auto-scaling with Kubernetes HPA
• Added Redis read-through caching for hot product listings

IN THIS PROJECT
Handled 5,000 requests/sec during peak harvest season."""

        parsed = answer_formatter.parse_structured_answer(raw, question_type="follow_up")
        titles = [s["title"] for s in parsed["sections"]]

        self.assertIn("SCALABILITY", titles)
        self.assertIn("IN THIS PROJECT", titles)

    # ============================================================
    # TEST 10: WIDGET RENDERING & PERFORMANCE BENCHMARK
    # ============================================================
    def test_10_rendering_and_performance(self):
        if not self.text_widget:
            self.skipTest("Tkinter display not available")

        sample_answer = """ANSWER
I implemented authentication using JWT tokens in Django REST Framework.

APPROACH
• User logs in with credentials
• Token is generated and signed
• Stored in HTTP-only secure cookie

WHY
• Stateless authentication
• Easy frontend integration

KEY POINT
Clean, secure stateless auth."""

        t_start = time.perf_counter()
        for _ in range(100):
            answer_formatter.render_structured_answer_to_text_widget(self.text_widget, sample_answer, "technical")
        t_elapsed_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

        print(f"\n[Answer Formatter Benchmark] Average parse & render latency: {t_elapsed_ms:.4f} ms")
        self.assertLess(t_elapsed_ms, 2.0, "Parsing and rendering must take < 2ms")

        # Verify text was actually inserted into widget
        content = self.text_widget.get("1.0", "end")
        self.assertIn("I implemented authentication", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
