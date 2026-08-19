"""
Run the Authoritative AI Copilot FastAPI Backend Server.
Usage:
    python run_backend.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import uvicorn
from backend.main import app

if __name__ == "__main__":
    print("===================================================================")
    print("🚀 Starting AI Copilot Authoritative Billing & Session Backend...")
    print("📍 URL: http://127.0.0.1:8000")
    print("📚 OpenAPI Docs: http://127.0.0.1:8000/docs")
    print("===================================================================")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
