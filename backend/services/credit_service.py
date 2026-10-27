from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.database.models import User, CreditTransaction


class CreditService:

    @staticmethod
    def deduct_credits_atomic(
        db: Session,
        user_id: str,
        amount_units: int,
        transaction_type: str,
        reference_id: Optional[str] = None,
        note: Optional[str] = None
    ) -> Tuple[bool, int, Optional[str]]:
        """
        Atomically deducts credit_units from a user ensuring no double-spending or negative balances.
        Returns: (success: bool, remaining_units: int, error_message: Optional[str])
        """
        if amount_units <= 0:
            return False, 0, "Deduction amount must be positive"

        # 1. Atomic SQL UPDATE with condition
        stmt = text("""
            UPDATE users
            SET credit_units = credit_units - :amount,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :user_id AND credit_units >= :amount
        """)
        result = db.execute(stmt, {"amount": amount_units, "user_id": user_id})

        if result.rowcount == 0:
            db.rollback()
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False, 0, "User not found"
            return False, user.credit_units, f"Insufficient credits: {user.credit_units} available, {amount_units} required"

        # 2. Fetch current balance after deduction (expire cache for raw update)
        db.expire_all()
        user = db.query(User).filter(User.id == user_id).first()
        remaining_units = user.credit_units if user else 0

        # 3. Create immutable ledger record
        tx = CreditTransaction(
            user_id=user_id,
            type=transaction_type,
            amount_units=-amount_units,
            balance_after_units=remaining_units,
            reference_id=reference_id,
            note=note
        )
        db.add(tx)
        db.commit()

        return True, remaining_units, None

    @staticmethod
    def add_credits_atomic(
        db: Session,
        user_id: str,
        amount_units: int,
        transaction_type: str,
        reference_id: Optional[str] = None,
        note: Optional[str] = None
    ) -> Tuple[bool, int, Optional[str]]:
        """
        Atomically adds credit_units to user and records transaction.
        """
        if amount_units <= 0:
            return False, 0, "Credit amount must be positive"

        stmt = text("""
            UPDATE users
            SET credit_units = credit_units + :amount,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :user_id
        """)

        result = db.execute(stmt, {"amount": amount_units, "user_id": user_id})
        if result.rowcount == 0:
            return False, 0, "User not found"

        db.expire_all()
        user = db.query(User).filter(User.id == user_id).first()
        remaining_units = user.credit_units if user else 0

        tx = CreditTransaction(
            user_id=user_id,
            type=transaction_type,
            amount_units=amount_units,
            balance_after_units=remaining_units,
            reference_id=reference_id,
            note=note
        )
        db.add(tx)
        db.commit()

        return True, remaining_units, None

    @staticmethod
    def get_ledger(db: Session, user_id: str, limit: int = 50) -> List[CreditTransaction]:
        return (
            db.query(CreditTransaction)
            .filter(CreditTransaction.user_id == user_id)
            .order_by(CreditTransaction.created_at.desc())
            .limit(limit)
            .all()
        )
