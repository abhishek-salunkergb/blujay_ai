import json
from fastapi import APIRouter, Request, Header, HTTPException, status, Depends
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.services.payment_service import PaymentService

router = APIRouter(prefix="/api/payments", tags=["Webhooks"])


@router.post("/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(None, alias="X-Razorpay-Signature"),
    db: Session = Depends(get_db)
):
    body_bytes = await request.body()
    
    # 1. Cryptographic Signature Verification
    if not PaymentService.verify_webhook_signature(body_bytes, x_razorpay_signature):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook cryptographic signature"
        )

    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = data.get("event")
    # We listen for order.paid or payment.captured
    if event_type in ["order.paid", "payment.captured"]:
        payload = data.get("payload", {}).get("payment", {}).get("entity", {})
        event_id = data.get("event_id") or payload.get("id")
        notes = payload.get("notes", {})
        user_id = notes.get("user_id")
        product_id = notes.get("product_id")
        payment_id = payload.get("id")
        order_id = payload.get("order_id")

        if user_id and product_id and event_id:
            success, msg, user = PaymentService.process_successful_payment(
                db=db,
                provider_event_id=event_id,
                user_id=user_id,
                product_id=product_id,
                payment_id=payment_id,
                order_id=order_id,
                raw_payload=data
            )
            return {"status": "ok", "message": msg}

    return {"status": "ignored", "event": event_type}
