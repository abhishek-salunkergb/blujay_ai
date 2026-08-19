import sys
import os
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base, User, Session as AppSession
from backend.services.session_service import SessionService
from backend.services.credit_service import CreditService

TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_session_lifecycle_and_30min_blocks():
    db = TestingSessionLocal()

    # Create user with 100 units (1 Credit = 2 Sessions)
    user = User(email="session_user@example.com", password_hash="hash", credit_units=100)
    db.add(user)
    db.commit()

    # 1. Start Session -> Deducts 50 units
    success, session, err, remaining = SessionService.start_session(db, user, mode="copilot")
    assert success is True
    assert remaining == 50
    assert session.status == "active"
    assert session.credit_units_used == 50
    assert session.expires_at > datetime.utcnow() + timedelta(minutes=28)

    # 2. Heartbeat -> Extends lease
    hb_success, hb_err, exp, lease_exp = SessionService.heartbeat(db, session.id, user.id)
    assert hb_success is True
    assert lease_exp > datetime.utcnow()

    # 3. Extend Session -> Deducts another 50 units (Remaining = 0)
    ext_success, ext_err, ext_remaining, new_exp = SessionService.extend_session(db, session.id, user)
    assert ext_success is True
    assert ext_remaining == 0
    assert new_exp > exp

    # 4. Attempt 2nd Extension -> Should Fail due to Insufficient Credits
    fail_ext, fail_err, fail_rem, _ = SessionService.extend_session(db, session.id, user)
    assert fail_ext is False
    assert "Insufficient credits" in (fail_err or "")

    # 5. End Session -> Completed
    end_success, _ = SessionService.end_session(db, session.id, user.id)
    assert end_success is True

    db.close()
    print("Session lifecycle test PASSED!")


def test_unlimited_subscription_bypass():
    db = TestingSessionLocal()

    # User with 0 units but active Unlimited pass
    unlimited_user = User(
        email="vip@example.com",
        password_hash="hash",
        credit_units=0,
        subscription_tier="unlimited",
        subscription_expires_at=datetime.utcnow() + timedelta(days=30)
    )
    db.add(unlimited_user)
    db.commit()

    # Should start without error and deduct 0 units
    success, session, err, remaining = SessionService.start_session(db, unlimited_user, mode="copilot")
    assert success is True
    assert remaining == 0
    assert session.credit_units_used == 0

    # Extension should also succeed with 0 deduction
    ext_success, _, ext_rem, _ = SessionService.extend_session(db, session.id, unlimited_user)
    assert ext_success is True
    assert ext_rem == 0

    db.close()
    print("Unlimited subscription test PASSED!")


if __name__ == "__main__":
    setup_module()
    test_session_lifecycle_and_30min_blocks()
    test_unlimited_subscription_bypass()
