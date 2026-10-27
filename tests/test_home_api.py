import sys
import os
import json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, User, Session as AppSession
from backend.database.connection import get_db
from backend.services.auth_service import create_access_token
from backend.main import app
from session_client import SessionClient

TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


def setup_module():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db


def teardown_module():
    app.dependency_overrides.clear()
    client.close()


def test_home_api_dashboard_and_stats():
    db = TestingSessionLocal()

    # 1. Create a test user
    user = User(
        email="home_tester@blujay.ai",
        password_hash="hashed_pw",
        credit_units=150,
        subscription_tier="pro_bundle"
    )
    db.add(user)
    db.commit()

    # 2. Add past mock sessions
    s1 = AppSession(
        user_id=user.id,
        session_mode="copilot",
        job_title="Senior AI Engineer",
        company_name="Anthropic / Stripe",
        status="completed",
        started_at=datetime.utcnow() - timedelta(hours=2),
        ended_at=datetime.utcnow() - timedelta(hours=1, minutes=30),
        expires_at=datetime.utcnow(),
        lease_expires_at=datetime.utcnow(),
        credit_units_used=50,
        qa_history_json=json.dumps([
            {"question": "Explain attention mechanism", "answer": "Scaled dot product attention computes similarity between queries and keys..."},
            {"question": "How do you handle Redis cache stampede?", "answer": "Using probabilistic early expiration (XFetch) or mutex locking..."}
        ]),
        summary_markdown="Candidate demonstrated strong knowledge of transformer models and distributed caching."
    )
    db.add(s1)
    db.commit()

    # Generate Auth Token
    token = create_access_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Test GET /api/home/dashboard
    resp = client.get("/api/home/dashboard", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["user"]["email"] == "home_tester@blujay.ai"
    assert data["user"]["credit_units"] == 150
    assert data["user"]["sessions_remaining"] == 3
    assert data["stats"]["total_sessions"] >= 1
    assert data["stats"]["total_questions_practiced"] >= 2
    assert len(data["recent_sessions"]) >= 1
    assert data["recent_sessions"][0]["job_title"] == "Senior AI Engineer"
    assert data["recent_sessions"][0]["total_questions"] == 2
    assert len(data["templates"]) >= 4
    assert data["system_status"]["api_status"] == "online"

    # 4. Test GET /api/home/stats
    stats_resp = client.get("/api/home/stats", headers=headers)
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert stats_data["total_sessions"] >= 1
    assert stats_data["total_questions_practiced"] >= 2

    # 5. Test GET /api/home/templates
    tpl_resp = client.get("/api/home/templates")
    assert tpl_resp.status_code == 200
    tpl_data = tpl_resp.json()
    assert len(tpl_data) >= 4
    template_ids = [t["id"] for t in tpl_data]
    assert "coding_leetcode" in template_ids
    assert "system_design" in template_ids
    assert "star_behavioral" in template_ids

    # 6. Test POST /api/home/quick-start
    qs_resp = client.post(
        "/api/home/quick-start",
        json={"template_id": "coding_leetcode", "custom_company": "Google"},
        headers=headers
    )
    assert qs_resp.status_code == 200
    qs_data = qs_resp.json()
    assert qs_data["success"] is True
    assert qs_data["company_name"] == "Google"
    assert qs_data["remaining_credit_units"] == 100  # 150 - 50 = 100

    db.close()


def test_session_client_offline_fallback():
    # Test that SessionClient gracefully returns complete dashboard fallback data even if server is offline
    offline_client = SessionClient(server_url="http://127.0.0.1:9999")
    dashboard = offline_client.fetch_home_dashboard()
    assert dashboard["success"] is True
    assert "user" in dashboard
    assert "stats" in dashboard
    assert "templates" in dashboard
    assert len(dashboard["templates"]) >= 4
    assert "recent_sessions" in dashboard


def test_home_api_analyze_fit():
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "home_tester@blujay.ai").first()
    if not user:
        user = User(
            email="home_tester@blujay.ai",
            password_hash="hashed_pw",
            credit_units=100,
            subscription_tier="pro_bundle"
        )
        db.add(user)
        db.commit()

    token = create_access_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Test with job description and resume
    jd_payload = {
        "job_description": "We are seeking a Senior Backend Engineer proficient in Python, FastAPI, Docker, and PostgreSQL to design scalable microservices.",
        "resume_text": "Experienced Python Engineer specializing in FastAPI, PostgreSQL, Redis, Docker, and microservices architecture.",
        "job_title": "Senior Backend Engineer"
    }
    resp = client.post("/api/home/analyze-fit", json=jd_payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert 0 <= data["match_score"] <= 100
    assert isinstance(data["matched_skills"], list)
    assert any(skill in [s.lower() for s in data["matched_skills"]] for skill in ["python", "fastapi", "docker", "postgresql"])
    assert isinstance(data["missing_skills"], list)
    assert isinstance(data["bridging_talking_points"], list)
    assert len(data["bridging_talking_points"]) > 0
    assert isinstance(data["predicted_questions"], list)
    assert len(data["predicted_questions"]) > 0

    # 2. Test empty JD edge case
    empty_resp = client.post("/api/home/analyze-fit", json={"job_description": ""}, headers=headers)
    assert empty_resp.status_code == 200
    empty_data = empty_resp.json()
    assert empty_data["success"] is True
    assert empty_data["match_score"] == 0
    assert empty_data["matched_skills"] == []

    db.close()

