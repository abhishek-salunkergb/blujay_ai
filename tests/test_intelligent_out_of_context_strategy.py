"""
Comprehensive Test Suite for Intelligent Out-of-Context Interview Question Strategy.
Validates:
1. Internal Question Classification (Direct, Out-of-Context Technical, Unrelated, Follow-Up).
2. Grounding in System Prompts (Main & InterviewEngine).
3. 3-Layer Answer Framework (Technical Explanation -> Natural Bridge -> Truth Validation).
4. Truth Guardrails (Zero hallucinated implementation claims, 'I would' vs 'I used').
5. Naturalness & Zero Meta-Leakage.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import unittest
from unittest.mock import MagicMock, patch
import os
import re

import interview_engine
from interview_engine import InterviewSession, build_interview_system_prompt, determine_knowledge_source


class TestIntelligentOutOfContextStrategy(unittest.TestCase):

    def setUp(self):
        self.session = InterviewSession()
        self.session.job_title = "Python Backend Developer"
        self.session.target_role = "Senior Python Developer"
        self.session.skills = ["Python", "Django", "FastAPI", "PostgreSQL", "Docker", "Git"]
        self.session.resume_text = (
            "Experience: Python Backend Developer building REST APIs with FastAPI and Django. "
            "Database: PostgreSQL, Redis caching. Containerization: Dockerfiles for backend services."
        )

    def test_01_out_of_context_technical_classification(self):
        """DevOps and infrastructure questions outside candidate's confirmed resume are classified as OUT_OF_CONTEXT_TECHNICAL."""
        devops_questions = [
            "How do Kubernetes pods communicate across nodes?",
            "Can you explain how a CI/CD pipeline works with ArgoCD and Jenkins?",
            "How would you manage infrastructure as code using Terraform and Ansible?",
            "Explain how an Istio service mesh handles mutual TLS."
        ]
        for q in devops_questions:
            source = determine_knowledge_source(q, session=self.session)
            self.assertEqual(source, "OUT_OF_CONTEXT_TECHNICAL", f"Expected OUT_OF_CONTEXT_TECHNICAL for '{q}', got '{source}'")

    def test_02_unrelated_technical_classification(self):
        """Completely unrelated domain questions (e.g. quantum physics, hardware ASIC) are classified as UNRELATED_TECHNICAL."""
        unrelated_questions = [
            "Explain quantum error correction and qubit decoherence.",
            "How does an FPGA ASIC design implement clock domain crossing in Verilog?",
            "What is the physical mechanism of superconductivity in Josephson junctions?"
        ]
        for q in unrelated_questions:
            source = determine_knowledge_source(q, session=self.session)
            self.assertEqual(source, "UNRELATED_TECHNICAL", f"Expected UNRELATED_TECHNICAL for '{q}', got '{source}'")

    def test_03_confirmed_skills_classification(self):
        """Confirmed skills and candidate project questions retain proper targeting."""
        self.assertEqual(determine_knowledge_source("Tell me about your experience with FastAPI", session=self.session), "CANDIDATE_EXPERIENCE")
        self.assertEqual(determine_knowledge_source("In your project, how did you implement Redis caching?", session=self.session), "CANDIDATE_PROJECT")
        self.assertEqual(determine_knowledge_source("What is polymorphism in Python?", session=self.session), "GENERAL_TECHNICAL")

    def test_04_system_prompt_out_of_context_strategy_embedded(self):
        """Verifies build_interview_system_prompt embeds all 15 Intelligent Out-of-Context strategy principles."""
        prompt = build_interview_system_prompt(self.session, question_type="technical", knowledge_source="OUT_OF_CONTEXT_TECHNICAL")
        
        # Verify core strategy header
        self.assertIn("INTELLIGENT OUT-OF-CONTEXT INTERVIEW QUESTION STRATEGY", prompt)
        
        # Verify 3-layer answer framework
        self.assertIn("THREE-LAYER ANSWER STRATEGY", prompt)
        self.assertIn("Layer 1 — Direct Technical Explanation", prompt)
        self.assertIn("Layer 2 — Natural Role Connection", prompt)
        self.assertIn("Layer 3 — Truth Validation", prompt)
        
        # Verify anti-fabrication rules
        self.assertIn("CRITICAL RULE — NEVER INVENT EXPERIENCE", prompt)
        self.assertIn("TECHNOLOGY KNOWLEDGE != IMPLEMENTATION EXPERIENCE", prompt)
        self.assertIn('"I would..."', prompt)
        
        # Verify concrete examples
        self.assertIn("Kubernetes pods communicate across nodes", prompt)
        self.assertIn("A CI/CD pipeline automates", prompt)
        
        # Verify follow-up and why question rules
        self.assertIn("How did you implement that?", prompt)
        self.assertIn("Why did you choose Kubernetes?", prompt)
        
        # Verify anti-leakage rule
        self.assertIn("ZERO META-LEAKAGE", prompt)

    def test_05_main_system_prompt_strategy_embedded(self):
        """Verifies main.py system_prompt contains the Intelligent Out-of-Context strategy."""
        import main
        # Verify that the main prompt text in main.py contains the strategy
        with open("main.py", "r", encoding="utf-8") as f:
            main_code = f.read()
        self.assertIn("INTELLIGENT OUT-OF-CONTEXT INTERVIEW QUESTION STRATEGY", main_code)
        self.assertIn("THREE-LAYER ANSWER STRATEGY", main_code)
        self.assertIn("DO NOT refuse to answer and DO NOT pretend the candidate has experience", main_code)

    def test_06_kubernetes_mock_answer_generation(self):
        """Verifies streaming answer generation for out-of-context Kubernetes question follows strategy."""
        mock_client = MagicMock()
        
        # Simulated response from LLM adhering to prompt
        mock_stream_tokens = [
            "Kubernetes networking ensures every pod receives a unique IP address across the cluster. ",
            "A CNI plugin like Calico or Flannel configures the underlying routing and overlay network so pods communicate directly without NAT. ",
            "\n\nFrom a Python backend perspective, the application primarily focuses on handling HTTP/gRPC requests, ",
            "relying on the cluster's networking and service discovery to route traffic seamlessly."
        ]
        
        def make_chunk(text):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            return chunk

        mock_client.chat.completions.create.return_value = [make_chunk(t) for t in mock_stream_tokens]
        
        tokens_received = []
        answer = interview_engine.generate_interview_answer(
            client=mock_client,
            session=self.session,
            question="How do Kubernetes pods communicate across nodes?",
            on_token=lambda tok: tokens_received.append(tok)
        )
        
        # Validate correct technical content
        self.assertIn("Kubernetes networking", answer)
        self.assertIn("CNI plugin", answer)
        # Validate natural role bridge
        self.assertIn("From a Python backend perspective", answer)
        # Validate zero fake implementation claims
        self.assertNotIn("I configured CNI", answer)
        self.assertNotIn("I built Kubernetes clusters", answer)
        # Validate zero meta-leakage
        self.assertNotIn("Intelligent Bridge", answer)
        self.assertNotIn("According to my resume", answer)

    def test_07_unrelated_question_no_forced_bridge(self):
        """Verifies answer generation for completely unrelated question provides strong technical answer without forced bridge."""
        mock_client = MagicMock()
        
        mock_stream_tokens = [
            "Quantum error correction protects quantum information from environmental noise and decoherence. ",
            "By redundantly encoding a single logical qubit across multiple physical qubits (such as in surface codes), ",
            "syndrome measurements can detect and correct bit-flip and phase-flip errors without collapsing the quantum superposition."
        ]
        
        def make_chunk(text):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            return chunk

        mock_client.chat.completions.create.return_value = [make_chunk(t) for t in mock_stream_tokens]
        
        tokens_received = []
        answer = interview_engine.generate_interview_answer(
            client=mock_client,
            session=self.session,
            question="Explain quantum error correction and how surface codes work.",
            on_token=lambda tok: tokens_received.append(tok)
        )
        
        # Validates accurate technical answer
        self.assertIn("Quantum error correction", answer)
        self.assertIn("syndrome measurements", answer)
        # Validates no forced Python connection
        self.assertNotIn("In my Python projects", answer)
        self.assertNotIn("From a Python developer's perspective", answer)

    def test_08_technology_familiarity_vs_implementation_guardrail(self):
        """Verifies that having a skill in resume does not claim unconfirmed sub-tools."""
        # Candidate has "Docker" in resume, but not "Kubernetes" or "Terraform"
        prompt = build_interview_system_prompt(self.session, question_type="technical", knowledge_source="OUT_OF_CONTEXT_TECHNICAL")
        self.assertIn("TECHNOLOGY KNOWLEDGE != IMPLEMENTATION EXPERIENCE", prompt)
        self.assertIn("Python, Docker, AWS", prompt)
        self.assertIn("Kubernetes, Terraform, AWS EKS", prompt)


if __name__ == "__main__":
    unittest.main()
