import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base, User, PaymentEvent
from backend.services.payment_service import PaymentService, PRODUCTS

TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_idempotent_payment_webhook_processing():
    db = TestingSessionLocal()

    # User with initial 100 units
    user = User(email="buyer@example.com", password_hash="hash", credit_units=100)
    db.add(user)
    db.commit()
    user_id = user.id

    event_id = "evt_razorpay_unique_99182"
    product_id = "pro_pack"  # +500 units

    # 1. First webhook delivery -> Should succeed & add +500 units (Total: 600)
    success1, msg1, updated_user = PaymentService.process_successful_payment(
        db=db,
        provider_event_id=event_id,
        user_id=user_id,
        product_id=product_id,
        payment_id="pay_12345",
        order_id="order_12345"
    )
    assert success1 is True
    assert updated_user.credit_units == 600

    # 2. Duplicate webhook delivery with same event_id -> Idempotent ignore (Balance stays 600)
    success2, msg2, same_user = PaymentService.process_successful_payment(
        db=db,
        provider_event_id=event_id,
        user_id=user_id,
        product_id=product_id,
        payment_id="pay_12345",
        order_id="order_12345"
    )
    assert success2 is True
    assert "already processed" in msg2.lower()
    
    # Verify DB still has 600 units (not 1100)
    check_user = db.query(User).filter(User.id == user_id).first()
    assert check_user.credit_units == 600

    # Verify only 1 PaymentEvent was recorded
    event_count = db.query(PaymentEvent).filter(PaymentEvent.provider_event_id == event_id).count()
    assert event_count == 1

    db.close()
    print("Payment webhook idempotency test PASSED!")


if __name__ == "__main__":
    setup_module()
    test_idempotent_payment_webhook_processing()
