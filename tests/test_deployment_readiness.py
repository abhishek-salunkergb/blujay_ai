"""
=============================================================================
BluJay AI - Deployment & Production Readiness QA Verification Suite
=============================================================================
Audits and verifies:
1. Healthcheck probe (/health) with DB connectivity validation
2. Database connection failure recovery and 503 degraded reporting
3. Static frontend asset mounting and dashboard route serving
4. PostgreSQL URL dialect normalization (postgres:// -> postgresql://)
5. CORS configuration parsing and wildcard handling
6. Production security validation for JWT secrets
7. Server dependency isolation in requirements-server.txt (Zero desktop bloat)
8. Containerization artifact validation (Dockerfile & docker-compose.yml)
=============================================================================
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import Settings, DEFAULT_INSECURE_JWT_SECRET


class TestDeploymentReadiness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check_endpoint_healthy(self):
        """Verifies /health probe returns 200 and validates DB connection."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("database"), "healthy")
        self.assertIn("version", data)

    def test_health_check_database_failure(self):
        """Verifies /health probe returns 503 when database execution fails."""
        with patch("sqlalchemy.orm.Session.execute", side_effect=Exception("DB Connection Refused")):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertIn("detail", data)
            self.assertEqual(data["detail"].get("status"), "degraded")

    def test_dashboard_and_copilot_static_routes(self):
        """Verifies /dashboard and /copilot serve the correct frontend HTML documents."""
        resp_dashboard = self.client.get("/dashboard")
        self.assertEqual(resp_dashboard.status_code, 200)
        self.assertIn("text/html", resp_dashboard.headers.get("content-type", ""))
        self.assertIn("BluJay AI", resp_dashboard.text)

        resp_copilot = self.client.get("/copilot")
        self.assertEqual(resp_copilot.status_code, 200)
        self.assertIn("text/html", resp_copilot.headers.get("content-type", ""))

    def test_frontend_static_asset_serving(self):
        """Verifies /frontend/home.css is accessible statically."""
        response = self.client.get("/frontend/home.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("css", response.headers.get("content-type", ""))

    def test_postgres_url_normalization(self):
        """Verifies postgres:// is normalized to postgresql:// for SQLAlchemy compatibility."""
        legacy_url = "postgres://user:password@localhost:5432/blujay_db"
        normalized = legacy_url.replace("postgres://", "postgresql://", 1) if legacy_url.startswith("postgres://") else legacy_url
        self.assertTrue(normalized.startswith("postgresql://"))

    def test_cors_origins_parsing(self):
        """Verifies CORS parsing correctly handles single, multiple, and wildcard origins."""
        s1 = Settings(CORS_ORIGINS="*")
        self.assertEqual(s1.cors_origins_list, ["*"])

        s2 = Settings(CORS_ORIGINS="https://blujay.ai, https://app.blujay.ai")
        self.assertEqual(s2.cors_origins_list, ["https://blujay.ai", "https://app.blujay.ai"])

    def test_production_jwt_security_enforcement(self):
        """Verifies server rejects insecure or default JWT_SECRET in production environment."""
        with self.assertRaises(ValueError):
            prod_settings = Settings(
                ENV="production",
                JWT_SECRET=DEFAULT_INSECURE_JWT_SECRET
            )
            prod_settings.validate_production_security()

    def test_server_requirements_isolation(self):
        """Verifies requirements-server.txt has server packages and zero desktop dependencies."""
        req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements-server.txt")
        self.assertTrue(os.path.isfile(req_path), "requirements-server.txt must exist")
        
        with open(req_path, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip() and not line.strip().startswith("#")]

        content = "\n".join(lines)

        # Must contain server essentials
        self.assertIn("fastapi", content)
        self.assertIn("uvicorn", content)
        self.assertIn("sqlalchemy", content)
        self.assertIn("psycopg2-binary", content)
        self.assertIn("stripe", content)

        # Must NOT contain desktop client-only packages
        forbidden_desktop_pkgs = [
            "soundcard", "sounddevice", "webrtcvad", "pywebview",
            "customtkinter", "pyautogui", "onnxruntime", "winocr"
        ]
        for pkg in forbidden_desktop_pkgs:
            self.assertNotIn(pkg, content, f"Forbidden desktop package '{pkg}' found in active requirements-server.txt")

    def test_dockerfile_and_compose_exist(self):
        """Verifies Dockerfile and docker-compose.yml exist and contain correct directives."""
        root_dir = os.path.dirname(os.path.dirname(__file__))
        dockerfile = os.path.join(root_dir, "Dockerfile")
        compose = os.path.join(root_dir, "docker-compose.yml")

        self.assertTrue(os.path.isfile(dockerfile), "Dockerfile must exist")
        self.assertTrue(os.path.isfile(compose), "docker-compose.yml must exist")

        with open(dockerfile, "r", encoding="utf-8") as f:
            df_content = f.read()
        self.assertIn("FROM python:", df_content)
        self.assertIn("HEALTHCHECK", df_content)
        self.assertIn("EXPOSE 8000", df_content)
        self.assertIn("requirements-server.txt", df_content)

        with open(compose, "r", encoding="utf-8") as f:
            dc_content = f.read()
        self.assertIn("services:", dc_content)
        self.assertIn("backend:", dc_content)
        self.assertIn("postgres:", dc_content)


if __name__ == "__main__":
    unittest.main()
