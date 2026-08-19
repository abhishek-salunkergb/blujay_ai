import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Copilot Authoritative Server"
    ENV: str = "development"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./copilot_server.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-copilot-production-key-change-in-env-9921")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days token

    # Payment Gateway (Razorpay & Stripe)
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "rzp_test_sample_key_123")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "sample_secret_key_456")
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "sample_webhook_secret_789")

    # Stripe Payment Gateway
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "sk_test_mock_stripe_key_123")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "pk_test_mock_stripe_pub_456")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_mock_stripe_webhook_secret_789")
    DEFAULT_CURRENCY: str = "usd"
    SUPPORTED_CURRENCIES: list = ["usd", "inr", "eur", "gbp"]

    # OAuth2 Social Login Providers
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "mock_google_client_id_123")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "mock_google_client_secret_456")
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "mock_github_client_id_123")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "mock_github_client_secret_456")
    OAUTH_REDIRECT_URI: str = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/api/auth/oauth/callback")
    REFRESH_TOKEN_EXPIRE_DAYS: int = 90

    # Admin & VIP Free Accounts (Whitelisted for permanent free unlimited access)
    ADMIN_EMAILS: str = os.getenv("ADMIN_EMAILS", "abhi958866@gmail.com,abhisheksalunke7387@gmail.com")

    # Business Rules
    INITIAL_CREDIT_UNITS: int = 100        # 1 Credit = 2 x 30m sessions
    SESSION_BLOCK_UNITS: int = 50          # 0.5 Credit = 50 units
    SESSION_BLOCK_MINUTES: int = 30        # 30-minute block lease
    SESSION_LEASE_GRACE_MINUTES: int = 2   # 2 minutes heartbeat lease

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
