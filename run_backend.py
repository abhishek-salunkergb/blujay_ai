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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print("===================================================================")
    print("🚀 Starting AI Copilot Authoritative Billing & Session Backend...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"📚 OpenAPI Docs: http://{host}:{port}/docs")
    print(f"🖥️ Web Dashboard: http://{host}:{port}/dashboard")
    print("===================================================================")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")
