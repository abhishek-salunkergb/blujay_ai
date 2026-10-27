"""
BluJay AI - Test Suite Package Initializer
Ensures the project root directory is always on sys.path for all test executions.
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
