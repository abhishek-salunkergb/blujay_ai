import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))
import concurrent.futures
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base, User
from backend.services.credit_service import CreditService

DB_FILE = "temp_concurrency_test.db"
if os.path.exists(DB_FILE):
    try:
        os.remove(DB_FILE)
    except Exception:
        pass

TEST_DB_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_concurrent_credit_deductions_prevent_double_spending():
    """
    CRITICAL TEST:
    A user has exactly 50 credit units.
    10 simultaneous threads attempt to deduct 50 units at the exact same moment.
    Expected: Exactly 1 thread succeeds; 9 threads fail. Final balance is exactly 0 units (never negative).
    """
    db = TestingSessionLocal()
    
    # 1. Create test user with exactly 50 units
    user = User(
        email="concurrent_test@example.com",
        password_hash="fake_hash",
        credit_units=50,
        subscription_tier="free"
    )
    db.add(user)
    db.commit()
    user_id = user.id
    db.close()

    results = []

    def attempt_deduction(thread_id):
        thread_db = TestingSessionLocal()
        try:
            success, remaining, err = CreditService.deduct_credits_atomic(
                db=thread_db,
                user_id=user_id,
                amount_units=50,
                transaction_type="session_start",
                note=f"Thread {thread_id} deduction"
            )
            return success, remaining, err
        finally:
            thread_db.close()

    # 2. Fire 10 concurrent threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_deduction, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    success_count = sum(1 for r in results if r[0] is True)
    fail_count = sum(1 for r in results if r[0] is False)

    # 3. Verify assertions
    assert success_count == 1, f"Expected exactly 1 successful deduction, got {success_count}"
    assert fail_count == 9, f"Expected 9 failed deductions, got {fail_count}"

    # 4. Check final user balance in DB
    verify_db = TestingSessionLocal()
    final_user = verify_db.query(User).filter(User.id == user_id).first()
    assert final_user.credit_units == 0, f"Expected final balance 0, got {final_user.credit_units}"
    verify_db.close()
    print("Concurrent deduction test PASSED cleanly!")


def teardown_module():
    engine.dispose()
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    setup_module()
    try:
        test_concurrent_credit_deductions_prevent_double_spending()
    finally:
        teardown_module()

