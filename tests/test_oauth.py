"""
Integration Test Suite for OAuth2 Authentication Pipeline (Feature 11)
======================================================================
Validates:
1. Google & GitHub authorization URL generation
2. Google OAuth callback, user upsert, and token issuance
3. GitHub OAuth callback, user upsert, and token issuance
4. JWT refresh token rotation (/api/auth/refresh)
5. Invalid/expired refresh token rejection
"""

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import Base, User
from backend.database.connection import get_db
from backend.services.auth_service import create_refresh_token
from backend.main import app

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


class TestOAuthPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        app.dependency_overrides[get_db] = override_get_db

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)

    def setUp(self):
        self.client = TestClient(app)
        self.db = TestingSessionLocal()

    def tearDown(self):
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()
        self.client.close()

    def test_01_google_authorize_url(self):
        """Verify Google authorize URL contains expected parameters."""
        res = self.client.get("/api/auth/oauth/google/authorize")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["provider"], "google")
        self.assertIn("accounts.google.com", data["authorize_url"])
        self.assertIn("state", data)

    def test_02_github_authorize_url(self):
        """Verify GitHub authorize URL contains expected parameters."""
        res = self.client.get("/api/auth/oauth/github/authorize")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["provider"], "github")
        self.assertIn("github.com/login/oauth/authorize", data["authorize_url"])
        self.assertIn("state", data)

    def test_03_google_oauth_callback_new_user(self):
        """Verify Google OAuth callback creates user and returns JWT tokens."""
        payload = {
            "code": "mock_google_code_99128",
            "state": "test_state_123"
        }
        res = self.client.post("/api/auth/oauth/google/callback", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)
        self.assertIn("user", data)
        self.assertEqual(data["user"]["credit_units"], 100)

        # Check DB user exists
        user = self.db.query(User).filter(User.email == data["user"]["email"]).first()
        self.assertIsNotNone(user)

    def test_04_github_oauth_callback(self):
        """Verify GitHub OAuth callback handles user creation."""
        payload = {
            "code": "mock_github_code_55412",
            "state": "test_state_456"
        }
        res = self.client.post("/api/auth/oauth/github/callback", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)

    def test_05_refresh_token_rotation(self):
        """Verify refreshing token produces new access token."""
        # 1. Create user
        user = User(
            email="refresh_test@example.com",
            password_hash="pw_hash_test",
            credit_units=100
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # 2. Issue refresh token
        ref_token = create_refresh_token(user_id=user.id, email=user.email)

        # 3. Call /refresh
        res = self.client.post("/api/auth/refresh", json={"refresh_token": ref_token})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)

    def test_06_invalid_refresh_token_rejected(self):
        """Verify invalid refresh token is rejected with 401."""
        res = self.client.post("/api/auth/refresh", json={"refresh_token": "invalid_refresh_token_123"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
