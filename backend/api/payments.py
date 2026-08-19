import uuid
import json
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import User, PaymentEvent
from backend.services.auth_service import get_current_user
from backend.services.payment_service import PaymentService, PRODUCTS
from backend.config import settings

router = APIRouter(prefix="/api/payments", tags=["Payments"])


class CreateOrderRequest(BaseModel):
    product_id: str


class CreateOrderResponse(BaseModel):
    order_id: str
    product_id: str
    amount_paise: int
    amount_inr: int
    currency: str = "INR"
    key_id: str
    product_name: str


class StripeCheckoutRequest(BaseModel):
    product_id: str
    currency: str = "usd"
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class MockCheckoutRequest(BaseModel):
    product_id: str


@router.get("/products")
@router.get("/catalog")
def get_products(currency: str = "usd"):
    """Returns multi-currency pricing catalog for subscription and credit packs."""
    return PaymentService.get_catalog(currency=currency)


@router.post("/create-order", response_model=CreateOrderResponse)
def create_order(req: CreateOrderRequest, user: User = Depends(get_current_user)):
    product = PRODUCTS.get(req.product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid product_id: {req.product_id}"
        )

    order_id = f"order_{uuid.uuid4().hex[:12]}"

    return CreateOrderResponse(
        order_id=order_id,
        product_id=product["id"],
        amount_paise=product["amount_paise"],
        amount_inr=product["price_inr"],
        key_id=settings.RAZORPAY_KEY_ID,
        product_name=product["name"]
    )


@router.post("/stripe/create-checkout-session")
def create_stripe_checkout(
    req: StripeCheckoutRequest,
    user: User = Depends(get_current_user)
):
    """
    Creates a Stripe Checkout Session for multi-currency payment.
    """
    try:
        session_data = PaymentService.create_stripe_checkout_session(
            user_id=user.id,
            user_email=user.email,
            product_id=req.product_id,
            currency=req.currency,
            success_url=req.success_url,
            cancel_url=req.cancel_url
        )
        return session_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe session creation error: {e}")


@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
    db: Session = Depends(get_db)
):
    """
    Authoritative Stripe Webhook endpoint.
    Verifies cryptographic webhook signature and idempotently credits user account.
    """
    payload_bytes = await request.body()

    # Cryptographic signature check
    if not PaymentService.verify_stripe_webhook_signature(payload_bytes, stripe_signature or ""):
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature")

    try:
        event = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = event.get("type", "")
    event_id = event.get("id", f"evt_{uuid.uuid4().hex[:10]}")
    data_object = event.get("data", {}).get("object", {})

    if event_type in ["checkout.session.completed", "payment_intent.succeeded"]:
        metadata = data_object.get("metadata", {})
        user_id = metadata.get("user_id") or data_object.get("client_reference_id")
        product_id = metadata.get("product_id", "starter_pack")
        payment_id = data_object.get("payment_intent") or data_object.get("id")

        if user_id:
            success, msg, _ = PaymentService.process_successful_payment(
                db=db,
                provider_event_id=event_id,
                user_id=user_id,
                product_id=product_id,
                payment_id=payment_id,
                raw_payload=event
            )
            return {"received": True, "processed": success, "message": msg}

    return {"received": True, "processed": False, "reason": f"Unhandled event {event_type}"}


@router.get("/transactions")
def get_transactions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns user's transaction and receipt history.
    """
    events = db.query(PaymentEvent).filter(PaymentEvent.user_id == user.id).order_by(PaymentEvent.created_at.desc()).all()
    return [
        {
            "id": e.id,
            "provider_event_id": e.provider_event_id,
            "product_id": e.product_id,
            "payment_id": e.payment_id,
            "order_id": e.order_id,
            "created_at": e.created_at.isoformat() if e.created_at else None
        }
        for e in events
    ]


@router.post("/mock-checkout")
def mock_checkout(
    req: MockCheckoutRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Test endpoint for immediate local development verification.
    Processes a verified checkout without needing external payment card.
    """
    mock_event_id = f"mock_evt_{uuid.uuid4().hex[:12]}"
    mock_payment_id = f"mock_pay_{uuid.uuid4().hex[:10]}"

    success, msg, updated_user = PaymentService.process_successful_payment(
        db=db,
        provider_event_id=mock_event_id,
        user_id=user.id,
        product_id=req.product_id,
        payment_id=mock_payment_id,
        order_id="mock_order_dev"
    )

    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "success": True,
        "message": msg,
        "credit_units": updated_user.credit_units,
        "credits_formatted": round(updated_user.credit_units / 100.0, 2),
        "subscription_tier": updated_user.subscription_tier,
        "is_unlimited": updated_user.has_unlimited_subscription()
    }
