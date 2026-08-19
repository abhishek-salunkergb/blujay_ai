from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import User
from backend.services.auth_service import get_current_user
from backend.services.credit_service import CreditService
from backend.config import settings

router = APIRouter(prefix="/api/user", tags=["Users & Credits"])


class CreditBalanceResponse(BaseModel):
    credit_units: int
    credits_formatted: float
    sessions_remaining: int
    subscription_tier: str
    is_unlimited: bool
    subscription_expires_at: str | None


class TransactionItem(BaseModel):
    id: str
    type: str
    amount_units: int
    amount_credits_formatted: float
    balance_after_units: int
    reference_id: str | None
    note: str | None
    created_at: str


@router.get("/credits", response_model=CreditBalanceResponse)
def get_credits(user: User = Depends(get_current_user)):
    is_unlimited = user.has_unlimited_subscription()
    return CreditBalanceResponse(
        credit_units=user.credit_units,
        credits_formatted=round(user.credit_units / 100.0, 2),
        sessions_remaining=user.credit_units // settings.SESSION_BLOCK_UNITS,
        subscription_tier=user.subscription_tier,
        is_unlimited=is_unlimited,
        subscription_expires_at=user.subscription_expires_at.isoformat() if user.subscription_expires_at else None
    )


@router.get("/transactions", response_model=List[TransactionItem])
def get_transactions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    txs = CreditService.get_ledger(db=db, user_id=user.id, limit=50)
    return [
        TransactionItem(
            id=t.id,
            type=t.type,
            amount_units=t.amount_units,
            amount_credits_formatted=round(t.amount_units / 100.0, 2),
            balance_after_units=t.balance_after_units,
            reference_id=t.reference_id,
            note=t.note,
            created_at=t.created_at.isoformat()
        )
        for t in txs
    ]
