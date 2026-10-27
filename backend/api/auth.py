from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import User
from backend.config import settings
from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    get_current_user
)
from backend.services.oauth_service import OAuthService

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class OAuthCallbackRequest(BaseModel):
    code: str
    state: Optional[str] = None
    redirect_uri: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    credit_units: int
    credits_formatted: float
    sessions_remaining: int
    subscription_tier: str
    subscription_expires_at: str | None
    is_unlimited: bool
    created_at: str


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user: UserResponse


def format_user_response(user: User) -> UserResponse:
    is_unlimited = user.has_unlimited_subscription()
    return UserResponse(
        id=user.id,
        email=user.email,
        credit_units=user.credit_units,
        credits_formatted=round(user.credit_units / 100.0, 2),
        sessions_remaining=user.credit_units // settings.SESSION_BLOCK_UNITS,
        subscription_tier=user.subscription_tier,
        subscription_expires_at=user.subscription_expires_at.isoformat() if user.subscription_expires_at else None,
        is_unlimited=is_unlimited,
        created_at=user.created_at.isoformat()
    )


@router.post("/register", response_model=AuthTokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    if not clean_email or len(req.password) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid email and password (min 4 characters) required."
        )

    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )

    # Create new user with initial 100 units (1.0 Credit = 2 Sessions)
    new_user = User(
        email=clean_email,
        password_hash=hash_password(req.password),
        credit_units=settings.INITIAL_CREDIT_UNITS,
        subscription_tier="free"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(user_id=new_user.id, email=new_user.email)
    refresh_tok = create_refresh_token(user_id=new_user.id, email=new_user.email)
    return AuthTokenResponse(
        access_token=token,
        refresh_token=refresh_tok,
        user=format_user_response(new_user)
    )


@router.post("/login", response_model=AuthTokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    token = create_access_token(user_id=user.id, email=user.email)
    refresh_tok = create_refresh_token(user_id=user.id, email=user.email)
    return AuthTokenResponse(
        access_token=token,
        refresh_token=refresh_tok,
        user=format_user_response(user)
    )


@router.get("/oauth/{provider}/authorize")
def oauth_authorize(provider: str, redirect_uri: Optional[str] = None):
    """
    Returns authorization URL and state token for Google or GitHub OAuth2.
    """
    try:
        return OAuthService.get_authorize_url(provider, redirect_uri=redirect_uri)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/oauth/{provider}/callback", response_model=AuthTokenResponse)
def oauth_callback(
    provider: str,
    req: OAuthCallbackRequest,
    db: Session = Depends(get_db)
):
    """
    Exchanges OAuth code for verified profile, upserts user, and returns JWT tokens.
    """
    try:
        profile = OAuthService.exchange_code_for_profile(
            provider=provider,
            code=req.code,
            redirect_uri=req.redirect_uri
        )
        email = profile.get("email")
        provider_uid = profile.get("provider_user_id", "")
        if not email:
            raise HTTPException(status_code=400, detail="Could not retrieve verified email from OAuth provider.")

        user, _ = OAuthService.get_or_create_oauth_user(
            db=db,
            email=email,
            provider=provider,
            provider_user_id=provider_uid
        )

        token = create_access_token(user_id=user.id, email=user.email)
        refresh_tok = create_refresh_token(user_id=user.id, email=user.email)

        return AuthTokenResponse(
            access_token=token,
            refresh_token=refresh_tok,
            user=format_user_response(user)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth login failed: {e}")


@router.post("/refresh")
def refresh_token(req: RefreshTokenRequest, db: Session = Depends(get_db)):
    """
    Refreshes expired access token using a valid refresh token.
    """
    payload = decode_access_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token."
        )

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    new_access_token = create_access_token(user_id=user.id, email=user.email)
    new_refresh_token = create_refresh_token(user_id=user.id, email=user.email)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "user": format_user_response(user)
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return format_user_response(current_user)
