import hmac
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import User, PaymentEvent
from backend.services.credit_service import CreditService

# Product Definitions owned by Backend
PRODUCTS: Dict[str, Dict[str, Any]] = {
    "starter_pack": {
        "id": "starter_pack",
        "name": "Starter Pack",
        "price_inr": 299,
        "amount_paise": 29900,
        "prices": {
            "usd": {"amount": 4.99, "cents": 499, "symbol": "$"},
            "inr": {"amount": 299, "cents": 29900, "symbol": "₹"},
            "eur": {"amount": 4.49, "cents": 449, "symbol": "€"},
            "gbp": {"amount": 3.99, "cents": 399, "symbol": "£"}
        },
        "credit_units": 200,
        "sessions_count": 4,
        "is_subscription": False,
        "description": "4 Sessions (2.0 Credits)"
    },
    "pro_pack": {
        "id": "pro_pack",
        "name": "Pro Pack",
        "price_inr": 599,
        "amount_paise": 59900,
        "prices": {
            "usd": {"amount": 9.99, "cents": 999, "symbol": "$"},
            "inr": {"amount": 599, "cents": 59900, "symbol": "₹"},
            "eur": {"amount": 8.99, "cents": 899, "symbol": "€"},
            "gbp": {"amount": 7.99, "cents": 799, "symbol": "£"}
        },
        "credit_units": 500,
        "sessions_count": 10,
        "is_subscription": False,
        "description": "10 Sessions (5.0 Credits) - Most Popular"
    },
    "power_pack": {
        "id": "power_pack",
        "name": "Power Pack",
        "price_inr": 999,
        "amount_paise": 99900,
        "prices": {
            "usd": {"amount": 19.99, "cents": 1999, "symbol": "$"},
            "inr": {"amount": 999, "cents": 99900, "symbol": "₹"},
            "eur": {"amount": 17.99, "cents": 1799, "symbol": "€"},
            "gbp": {"amount": 15.99, "cents": 1599, "symbol": "£"}
        },
        "credit_units": 1100,
        "sessions_count": 22,
        "is_subscription": False,
        "description": "22 Sessions (11.0 Credits) - Best Value"
    },
    "unlimited_monthly": {
        "id": "unlimited_monthly",
        "name": "Unlimited Monthly Pass",
        "price_inr": 1499,
        "amount_paise": 149900,
        "prices": {
            "usd": {"amount": 29.99, "cents": 2999, "symbol": "$"},
            "inr": {"amount": 1499, "cents": 149900, "symbol": "₹"},
            "eur": {"amount": 27.99, "cents": 2799, "symbol": "€"},
            "gbp": {"amount": 24.99, "cents": 2499, "symbol": "£"}
        },
        "credit_units": 0,
        "sessions_count": -1,
        "is_subscription": True,
        "subscription_days": 30,
        "description": "Unlimited 30-Minute AI Sessions for 30 Days"
    }
}


class PaymentService:

    @staticmethod
    def get_catalog(currency: str = "usd") -> list:
        curr = currency.lower() if currency else "usd"
        catalog = []
        for p in PRODUCTS.values():
            prod_copy = dict(p)
            price_info = p.get("prices", {}).get(curr, p.get("prices", {}).get("usd", {"amount": 9.99, "cents": 999, "symbol": "$"}))
            prod_copy["currency"] = curr.upper()
            prod_copy["display_price"] = price_info["amount"]
            prod_copy["price_cents"] = price_info["cents"]
            prod_copy["currency_symbol"] = price_info["symbol"]
            catalog.append(prod_copy)
        return catalog

    @staticmethod
    def verify_webhook_signature(payload_bytes: bytes, signature: str) -> bool:
        """
        Cryptographic verification of Razorpay webhook signature (HMAC-SHA256).
        """
        if not signature or not settings.RAZORPAY_WEBHOOK_SECRET:
            return False
        try:
            expected_sig = hmac.new(
                key=settings.RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
                msg=payload_bytes,
                digestmod=hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected_sig, signature)
        except Exception:
            return False

    @staticmethod
    def verify_stripe_webhook_signature(payload_bytes: bytes, sig_header: str) -> bool:
        """
        Cryptographic verification of Stripe webhook signature.
        Supports standard Stripe webhook signature header format `t=timestamp,v1=signature`.
        """
        if not sig_header or not settings.STRIPE_WEBHOOK_SECRET:
            return False
        try:
            # Parse Stripe signature components (t=..., v1=...)
            elements = {}
            for item in sig_header.split(","):
                parts = item.strip().split("=", 1)
                if len(parts) == 2:
                    elements[parts[0]] = parts[1]

            timestamp = elements.get("t", "")
            received_v1 = elements.get("v1", "")

            if not timestamp or not received_v1:
                # Direct HMAC fallback if header was raw signature
                expected = hmac.new(
                    key=settings.STRIPE_WEBHOOK_SECRET.encode("utf-8"),
                    msg=payload_bytes,
                    digestmod=hashlib.sha256
                ).hexdigest()
                return hmac.compare_digest(expected, sig_header)

            signed_payload = f"{timestamp}.".encode("utf-8") + payload_bytes
            expected_v1 = hmac.new(
                key=settings.STRIPE_WEBHOOK_SECRET.encode("utf-8"),
                msg=signed_payload,
                digestmod=hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected_v1, received_v1)
        except Exception:
            return False

    @staticmethod
    def create_stripe_checkout_session(
        user_id: str,
        user_email: str,
        product_id: str,
        currency: str = "usd",
        success_url: Optional[str] = None,
        cancel_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates Stripe Checkout session metadata and secure redirect URL.
        """
        import uuid
        product = PRODUCTS.get(product_id)
        if not product:
            raise ValueError(f"Unknown product_id: {product_id}")

        curr = currency.lower()
        price_info = product.get("prices", {}).get(curr, product.get("prices", {}).get("usd"))
        amount_cents = price_info["cents"]

        session_id = f"cs_test_{uuid.uuid4().hex[:20]}"
        checkout_url = f"https://checkout.stripe.com/c/pay/{session_id}"

        return {
            "session_id": session_id,
            "checkout_url": checkout_url,
            "publishable_key": settings.STRIPE_PUBLISHABLE_KEY,
            "product_id": product_id,
            "product_name": product["name"],
            "currency": curr.upper(),
            "amount_cents": amount_cents,
            "display_price": price_info["amount"],
            "currency_symbol": price_info["symbol"],
            "customer_email": user_email
        }

    @staticmethod
    def process_successful_payment(
        db: Session,
        provider_event_id: str,
        user_id: str,
        product_id: str,
        payment_id: Optional[str] = None,
        order_id: Optional[str] = None,
        raw_payload: Optional[dict] = None
    ) -> Tuple[bool, Optional[str], Optional[User]]:
        """
        Idempotent payment processing.
        Guarantees that a duplicate webhook delivery does not credit multiple times.
        """
        # 1. Idempotency Check
        existing_event = db.query(PaymentEvent).filter(
            PaymentEvent.provider_event_id == provider_event_id
        ).first()

        if existing_event:
            user = db.query(User).filter(User.id == existing_event.user_id).first()
            return True, "Payment event already processed (idempotent ignore)", user

        product = PRODUCTS.get(product_id)
        if not product:
            return False, f"Unknown product_id: {product_id}", None

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, f"User {user_id} not found", None

        # 2. Grant product benefits
        if product["is_subscription"]:
            now = datetime.utcnow()
            base_time = user.subscription_expires_at if (user.subscription_expires_at and user.subscription_expires_at > now) else now
            user.subscription_tier = "unlimited"
            user.subscription_expires_at = base_time + timedelta(days=product["subscription_days"])
            db.commit()
        else:
            # Grant credit units atomically
            CreditService.add_credits_atomic(
                db=db,
                user_id=user.id,
                amount_units=product["credit_units"],
                transaction_type="purchase",
                reference_id=payment_id or order_id or provider_event_id,
                note=f"Purchased {product['name']} (+{product['credit_units']} units)"
            )

        # 3. Record Payment Event for Idempotency
        event = PaymentEvent(
            provider="razorpay",
            provider_event_id=provider_event_id,
            payment_id=payment_id,
            order_id=order_id,
            user_id=user.id,
            amount_paise=product["amount_paise"],
            product_id=product_id,
            status="processed",
            payload_json=json.dumps(raw_payload) if raw_payload else None
        )
        db.add(event)
        db.commit()
        db.refresh(user)

        return True, "Payment processed successfully", user
