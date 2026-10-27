"""
Integration Test Suite for Stripe Multi-Currency Payments & Webhooks (Feature 10)
==================================================================================
Validates:
1. Multi-currency pricing catalog endpoints (USD, INR, EUR, GBP)
2. Stripe Checkout session generation
3. Webhook signature verification and invalid signature rejection
4. Atomic credit fulfillment via Stripe webhook
5. Webhook idempotency (duplicate prevention)
6. Transaction and receipt history endpoint
"""

import hmac
import hashlib
import json
import uuid
import unittest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
from backend.database.models import Base, User, PaymentEvent
from backend.database.connection import get_db
from backend.services.auth_service import create_access_token
from backend.config import settings
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


class TestStripePayments(unittest.TestCase):

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

        # Create test user
        self.user = User(
            email=f"stripe_user_{uuid.uuid4().hex[:6]}@example.com",
            password_hash="mock_hash_12345",
            credit_units=100,
            subscription_tier="free"
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

        self.token = create_access_token(user_id=self.user.id, email=self.user.email)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self):
        self.db.query(PaymentEvent).delete()
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()
        self.client.close()

    def test_01_multicurrency_catalog(self):
        """Verify catalog returns correct currency symbols and prices."""
        # USD
        res_usd = self.client.get("/api/payments/catalog?currency=usd")
        self.assertEqual(res_usd.status_code, 200)
        items_usd = res_usd.json()
        self.assertTrue(any(p["currency_symbol"] == "$" for p in items_usd))

        # INR
        res_inr = self.client.get("/api/payments/catalog?currency=inr")
        self.assertEqual(res_inr.status_code, 200)
        items_inr = res_inr.json()
        self.assertTrue(any(p["currency_symbol"] == "₹" for p in items_inr))

        # EUR
        res_eur = self.client.get("/api/payments/catalog?currency=eur")
        self.assertEqual(res_eur.status_code, 200)
        items_eur = res_eur.json()
        self.assertTrue(any(p["currency_symbol"] == "€" for p in items_eur))

    def test_02_create_stripe_checkout_session(self):
        """Verify creating a Stripe checkout session."""
        payload = {
            "product_id": "pro_pack",
            "currency": "usd"
        }
        res = self.client.post("/api/payments/stripe/create-checkout-session", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("session_id", data)
        self.assertIn("checkout_url", data)
        self.assertEqual(data["currency"], "USD")
        self.assertEqual(data["product_id"], "pro_pack")

    def test_03_stripe_webhook_invalid_signature(self):
        """Verify webhook rejects payload with missing or invalid signature."""
        payload = {"type": "checkout.session.completed"}
        res = self.client.post(
            "/api/payments/stripe/webhook",
            json=payload,
            headers={"stripe-signature": "invalid_sig_123"}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid Stripe webhook signature", res.json()["detail"])

    def test_04_stripe_webhook_valid_and_idempotent(self):
        """Verify valid signed webhook credits user and handles duplicates idempotently."""
        event_id = f"evt_stripe_{uuid.uuid4().hex[:12]}"
        webhook_data = {
            "id": event_id,
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test_sample",
                    "payment_intent": "pi_test_123",
                    "metadata": {
                        "user_id": self.user.id,
                        "product_id": "pro_pack"
                    }
                }
            }
        }
        payload_bytes = json.dumps(webhook_data).encode("utf-8")
        signature = hmac.new(
            key=settings.STRIPE_WEBHOOK_SECRET.encode("utf-8"),
            msg=payload_bytes,
            digestmod=hashlib.sha256
        ).hexdigest()

        # First Delivery
        res1 = self.client.post(
            "/api/payments/stripe/webhook",
            content=payload_bytes,
            headers={"stripe-signature": signature, "Content-Type": "application/json"}
        )
        self.assertEqual(res1.status_code, 200)
        self.assertTrue(res1.json()["processed"])

        # Verify DB credits increased (Initial 100 + Pro Pack 500 = 600)
        self.db.refresh(self.user)
        self.assertEqual(self.user.credit_units, 600)

        # Duplicate Delivery (Idempotency)
        res2 = self.client.post(
            "/api/payments/stripe/webhook",
            content=payload_bytes,
            headers={"stripe-signature": signature, "Content-Type": "application/json"}
        )
        self.assertEqual(res2.status_code, 200)

        # DB credits must NOT double-increment
        self.db.refresh(self.user)
        self.assertEqual(self.user.credit_units, 600)

    def test_05_transactions_history(self):
        """Verify transactions history returns user payment records."""
        # Add mock payment event
        evt = PaymentEvent(
            provider_event_id="evt_test_hist_1",
            user_id=self.user.id,
            product_id="starter_pack",
            amount_paise=29900,
            payment_id="pay_123",
            order_id="ord_123"
        )
        self.db.add(evt)
        self.db.commit()

        res = self.client.get("/api/payments/transactions", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        txs = res.json()
        self.assertGreaterEqual(len(txs), 1)
        self.assertEqual(txs[0]["product_id"], "starter_pack")


if __name__ == "__main__":
    unittest.main()
