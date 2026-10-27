
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import unittest
from dotenv import load_dotenv

load_dotenv()

class TestModelsConfiguration(unittest.TestCase):
    def test_groq_model_connectivity(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.skipTest("GROQ_API_KEY not configured")
            
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            self.assertIsNotNone(client)
        except Exception as e:
            self.fail(f"Failed to initialize Groq client: {e}")

if __name__ == "__main__":
    unittest.main()

