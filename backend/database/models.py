import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # Integer accounting: 1 Credit = 100 units; 0.5 Credit = 50 units
    credit_units = Column(Integer, default=100, nullable=False)
    
    # Subscriptions: "free", "pay_as_you_go", "unlimited", "admin", "lifetime_free"
    subscription_tier = Column(String(50), default="free", nullable=False)
    subscription_expires_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("CreditTransaction", back_populates="user", cascade="all, delete-orphan")
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")

    def has_unlimited_subscription(self) -> bool:
        from backend.config import settings

        # 1. Check permanent Admin/VIP tiers or whitelist
        if self.subscription_tier in ["admin", "lifetime_free", "vip"]:
            return True

        admin_list = getattr(settings, "admin_emails_list", [])
        if "*" in admin_list or (self.email and self.email.lower() in admin_list):
            return True

        # 2. Check time-based unlimited subscription
        if self.subscription_tier == "unlimited" and self.subscription_expires_at:
            return self.subscription_expires_at > datetime.utcnow()
        return False


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    
    session_mode = Column(String(50), default="copilot", nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)              # 30-min block boundary
    last_heartbeat_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    lease_expires_at = Column(DateTime, nullable=False)       # 2-minute lease for crash cleanup
    ended_at = Column(DateTime, nullable=True)
    
    credit_units_used = Column(Integer, default=50, nullable=False)
    
    # Statuses: "active", "completed", "expired", "insufficient_funds", "cancelled"
    status = Column(String(50), default="active", index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Post-interview persistent history & analytics
    job_title = Column(String(255), nullable=True)
    company_name = Column(String(255), nullable=True)
    qa_history_json = Column(Text, nullable=True)
    transcript_json = Column(Text, nullable=True)
    summary_markdown = Column(Text, nullable=True)

    user = relationship("User", back_populates="sessions")


class CreditTransaction(Base):
    __tablename__ = "credit_transactions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    
    # Types: "purchase", "session_start", "session_extension", "refund", "bonus", "admin_adjustment"
    type = Column(String(50), nullable=False)
    amount_units = Column(Integer, nullable=False)          # e.g. +500, -50
    balance_after_units = Column(Integer, nullable=False)
    reference_id = Column(String(255), nullable=True)       # e.g. session_id or payment_id
    note = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="transactions")


class PaymentEvent(Base):
    __tablename__ = "payment_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    provider = Column(String(50), default="razorpay", nullable=False)
    provider_event_id = Column(String(255), unique=True, index=True, nullable=False)
    payment_id = Column(String(255), index=True, nullable=True)
    order_id = Column(String(255), index=True, nullable=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    amount_paise = Column(Integer, nullable=False)           # INR in paise (₹299 = 29900)
    product_id = Column(String(50), nullable=False)
    status = Column(String(50), default="processed", nullable=False)
    payload_json = Column(Text, nullable=True)
    
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Device(Base):
    __tablename__ = "devices"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    device_id = Column(String(255), nullable=False)
    device_name = Column(String(255), nullable=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="devices")
