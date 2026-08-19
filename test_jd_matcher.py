"""
Unit Test Suite for Job Description Matcher & Fit Analysis Engine (Feature 5)
=============================================================================
Validates:
1. Heuristic keyword extraction from tech text
2. Semantic matching and score computation between JD and Resume
3. Missing skill gap detection and bridging narrative generation
4. High-probability interview question prediction
5. LLM JSON output parsing and fallback isolation
6. Local analysis caching round-trips
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import jd_matcher
from jd_matcher import (
    extract_keywords_heuristic,
    heuristic_jd_match,
    analyze_jd_resume_fit,
    save_jd_analysis_cache,
    get_cached_jd_analysis
)


class TestJDMatcher(unittest.TestCase):

    def setUp(self):
        self.temp_cache = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_cache.close()
        self._orig_cache = jd_matcher.JD_ANALYSIS_CACHE_FILE
        jd_matcher.JD_ANALYSIS_CACHE_FILE = self.temp_cache.name

    def tearDown(self):
        jd_matcher.JD_ANALYSIS_CACHE_FILE = self._orig_cache
        if os.path.exists(self.temp_cache.name):
            try:
                os.remove(self.temp_cache.name)
            except Exception:
                pass

    def test_01_extract_keywords_heuristic(self):
        """Verify technical keyword extractor identifies core languages and platforms."""
        text = "We are seeking a Senior Python & React engineer experienced with Docker, AWS, and PostgreSQL."
        kw = extract_keywords_heuristic(text)
        self.assertIn("python", kw)
        self.assertIn("react", kw)
        self.assertIn("docker", kw)
        self.assertIn("aws", kw)
        self.assertIn("postgresql", kw)

    def test_02_heuristic_jd_match_high_overlap(self):
        """Verify match calculation with significant skill overlap."""
        jd = "Looking for a Python, FastAPI, Docker, and PostgreSQL developer."
        resume = "Experienced software engineer with Python, FastAPI, Docker, PostgreSQL, and Redis."
        
        result = heuristic_jd_match(jd, resume)
        self.assertGreaterEqual(result["match_score"], 80)
        self.assertIn("python", result["matched_skills"])
        self.assertIn("fastapi", result["matched_skills"])
        self.assertGreaterEqual(len(result["predicted_questions"]), 1)

    def test_03_heuristic_jd_match_with_gaps(self):
        """Verify missing skills and bridging talking points are computed when gaps exist."""
        jd = "Requires Rust, Kubernetes, Kafka, and GraphQL."
        resume = "Built APIs using Python, Flask, and Docker."
        
        result = heuristic_jd_match(jd, resume)
        self.assertIn("rust", result["missing_skills"])
        self.assertIn("kubernetes", result["missing_skills"])
        self.assertGreaterEqual(len(result["bridging_talking_points"]), 1)

    def test_04_analyze_jd_resume_fit_empty_jd(self):
        """Verify empty JD returns score 0 and friendly prompt."""
        result = analyze_jd_resume_fit("", "Some resume text")
        self.assertEqual(result["match_score"], 0)
        self.assertEqual(len(result["matched_skills"]), 0)

    def test_05_analyze_jd_resume_fit_llm_success(self):
        """Verify analyze_jd_resume_fit parses LLM JSON structured response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "match_score": 92,
                "matched_skills": ["Python", "System Design", "AWS"],
                "missing_skills": ["Terraform"],
                "bridging_talking_points": ["Discuss CloudFormation experience as bridge to Terraform."],
                "predicted_questions": ["Explain how you design multi-region failovers on AWS."]
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        res = analyze_jd_resume_fit("Senior Backend Role on AWS", "AWS & Python architect", client=mock_client)
        self.assertEqual(res["match_score"], 92)
        self.assertIn("Python", res["matched_skills"])
        self.assertIn("Terraform", res["missing_skills"])

    def test_06_cache_persistence_roundtrip(self):
        """Verify saving and retrieving cached analysis from disk."""
        data = {
            "match_score": 85,
            "matched_skills": ["Go", "Docker"],
            "missing_skills": ["Kubernetes"],
            "bridging_talking_points": ["Show containerization depth."],
            "predicted_questions": ["How do you handle goroutine leaks?"]
        }
        save_jd_analysis_cache(data)
        loaded = get_cached_jd_analysis()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["match_score"], 85)
        self.assertEqual(loaded["matched_skills"], ["Go", "Docker"])


if __name__ == "__main__":
    unittest.main()
