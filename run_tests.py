"""
BluJay AI - Unified Test Runner
================================
Run test suites cleanly from project root without configuration hassles.

Usage:
    python run_tests.py              # Runs standard test suite
    python run_tests.py --suite fast # Runs fast logic/hardening tests
    python run_tests.py --suite all  # Runs full test discovery
"""

import sys
import os
import unittest
import argparse

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

FAST_TESTS = [
    "tests.test_parakeet_followup_hardening",
    "tests.test_phase1_followup_and_topic",
    "tests.test_category5_rag_grounding",
    "tests.test_wiring_and_disconnections",
    "tests.test_credits_concurrency",
    "tests.test_home_api",
]

def run_suite(suite_name="fast"):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if suite_name == "fast":
        print(f"--> Running Fast Hardening & Logic Test Suite ({len(FAST_TESTS)} modules)...")
        for mod in FAST_TESTS:
            try:
                suite.addTests(loader.loadTestsFromName(mod))
            except Exception as e:
                print(f"[WARN] Failed loading {mod}: {e}")
    else:
        print("--> Discovering all tests under tests/...")
        suite = loader.discover(start_dir=os.path.join(_PROJECT_ROOT, "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BluJay Test Runner")
    parser.add_argument("--suite", choices=["fast", "all"], default="fast", help="Suite to run (default: fast)")
    args = parser.parse_args()
    run_suite(args.suite)
