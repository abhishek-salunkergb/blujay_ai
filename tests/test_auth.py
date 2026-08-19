import sys
import os
sys.path.insert(0, os.path.abspath("."))
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from backend.database.models import Base, User
from backend.services.auth_service import hash_password, verify_password, create_access_token, decode_access_token

TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_password_hashing_and_verification():
    raw_pass = "SecurePass123!"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPass", hashed) is False
    print("Password hashing test PASSED!")


def test_jwt_token_generation_and_decode():
    user_id = "test-uuid-1234"
    email = "test@ai-copilot.com"
    token = create_access_token(user_id=user_id, email=email)
    assert isinstance(token, str)
    
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == user_id
    assert payload["email"] == email
    print("JWT token test PASSED!")


if __name__ == "__main__":
    setup_module()
    test_password_hashing_and_verification()
    test_jwt_token_generation_and_decode()
