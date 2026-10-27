import os
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_INSECURE_JWT_SECRET = "super-secret-copilot-production-key-change-in-env-9921"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    APP_NAME: str = "AI Copilot Authoritative Server"
    ENV: str = os.getenv("ENV", "development")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./copilot_server.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET") or os.getenv("JWT_SECRET_KEY") or DEFAULT_INSECURE_JWT_SECRET
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
    ADMIN_EMAILS: str = os.getenv("ADMIN_EMAILS", "")

    # CORS Whitelist
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # Business Rules
    INITIAL_CREDIT_UNITS: int = 100        # 1 Credit = 2 x 30m sessions
    SESSION_BLOCK_UNITS: int = 50          # 0.5 Credit = 50 units
    SESSION_BLOCK_MINUTES: int = 30        # 30-minute block lease
    SESSION_LEASE_GRACE_MINUTES: int = 2   # 2 minutes heartbeat lease

    @property
    def cors_origins_list(self) -> list:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def admin_emails_list(self) -> list:
        return [e.strip().lower() for e in self.ADMIN_EMAILS.split(",") if e.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENV.strip().lower() in ("production", "prod")

    def validate_production_security(self):
        if self.is_production:
            if not self.JWT_SECRET or self.JWT_SECRET == DEFAULT_INSECURE_JWT_SECRET:
                raise ValueError(
                    "CRITICAL SECURITY ERROR: JWT_SECRET must be explicitly set to a "
                    "cryptographically secure random string in production environment."
                )


settings = Settings()
settings.validate_production_security()

