from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session as DBSession
from backend.config import settings
from backend.database.models import User, Session as AppSession
from backend.services.credit_service import CreditService


class SessionService:

    @staticmethod
    def start_session(
        db: DBSession,
        user: User,
        mode: str = "copilot"
    ) -> Tuple[bool, Optional[AppSession], Optional[str], int]:
        """
        Starts a 30-minute session block.
        Deducts 50 credit_units unless the user holds an active Unlimited subscription.
        Returns: (success: bool, session: Optional[AppSession], error_msg: Optional[str], remaining_units: int)
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=settings.SESSION_BLOCK_MINUTES)
        lease_expires = now + timedelta(minutes=settings.SESSION_LEASE_GRACE_MINUTES)

        # 1. Check for unlimited subscription
        is_unlimited = user.has_unlimited_subscription()
        units_to_deduct = 0 if is_unlimited else settings.SESSION_BLOCK_UNITS

        if not is_unlimited:
            success, remaining, error = CreditService.deduct_credits_atomic(
                db=db,
                user_id=user.id,
                amount_units=units_to_deduct,
                transaction_type="session_start",
                note=f"Start 30-min {mode} session"
            )
            if not success:
                return False, None, error, remaining
        else:
            remaining = user.credit_units

        # 2. Create the Session record
        session = AppSession(
            user_id=user.id,
            session_mode=mode,
            started_at=now,
            expires_at=expires_at,
            last_heartbeat_at=now,
            lease_expires_at=lease_expires,
            credit_units_used=units_to_deduct,
            status="active"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        return True, session, None, remaining

    @staticmethod
    def heartbeat(
        db: DBSession,
        session_id: str,
        user_id: str
    ) -> Tuple[bool, Optional[str], Optional[datetime], Optional[datetime]]:
        """
        Refreshes the 2-minute lease for an active session.
        Returns: (success: bool, error: Optional[str], expires_at, lease_expires_at)
        """
        session = db.query(AppSession).filter(
            AppSession.id == session_id,
            AppSession.user_id == user_id
        ).first()

        if not session:
            return False, "Session not found", None, None

        if session.status != "active":
            return False, f"Session is {session.status}", session.expires_at, session.lease_expires_at

        now = datetime.utcnow()

        # If block expiration reached without extension
        if now > session.expires_at:
            session.status = "expired"
            session.ended_at = now
            db.commit()
            return False, "Session 30-minute block expired", session.expires_at, session.lease_expires_at

        session.last_heartbeat_at = now
        session.lease_expires_at = now + timedelta(minutes=settings.SESSION_LEASE_GRACE_MINUTES)
        db.commit()

        return True, None, session.expires_at, session.lease_expires_at

    @staticmethod
    def extend_session(
        db: DBSession,
        session_id: str,
        user: User
    ) -> Tuple[bool, Optional[str], int, Optional[datetime]]:
        """
        Extends the active session by another 30-minute block (-50 units unless unlimited).
        Returns: (success: bool, error_msg: Optional[str], remaining_units: int, new_expires_at: Optional[datetime])
        """
        session = db.query(AppSession).filter(
            AppSession.id == session_id,
            AppSession.user_id == user.id
        ).first()

        if not session:
            return False, "Session not found", user.credit_units, None

        if session.status != "active":
            return False, f"Cannot extend {session.status} session", user.credit_units, session.expires_at

        is_unlimited = user.has_unlimited_subscription()
        units_to_deduct = 0 if is_unlimited else settings.SESSION_BLOCK_UNITS

        if not is_unlimited:
            success, remaining, error = CreditService.deduct_credits_atomic(
                db=db,
                user_id=user.id,
                amount_units=units_to_deduct,
                transaction_type="session_extension",
                reference_id=session.id,
                note="Auto-extend session +30 min"
            )
            if not success:
                return False, error, remaining, session.expires_at
        else:
            remaining = user.credit_units

        # Extend expiration by 30 minutes from current expires_at
        now = datetime.utcnow()
        base_time = max(session.expires_at, now)
        session.expires_at = base_time + timedelta(minutes=settings.SESSION_BLOCK_MINUTES)
        session.lease_expires_at = now + timedelta(minutes=settings.SESSION_LEASE_GRACE_MINUTES)
        session.credit_units_used += units_to_deduct
        db.commit()

        return True, None, remaining, session.expires_at

    @staticmethod
    def end_session(
        db: DBSession,
        session_id: str,
        user_id: str,
        qa_history_json: Optional[str] = None,
        transcript_json: Optional[str] = None,
        summary_markdown: Optional[str] = None,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Terminates the session, marks it completed, and persists final transcripts and analytics.
        """
        session = db.query(AppSession).filter(
            AppSession.id == session_id,
            AppSession.user_id == user_id
        ).first()

        if not session:
            return False, "Session not found"

        session.status = "completed"
        session.ended_at = datetime.utcnow()
        if qa_history_json is not None:
            session.qa_history_json = qa_history_json
        if transcript_json is not None:
            session.transcript_json = transcript_json
        if summary_markdown is not None:
            session.summary_markdown = summary_markdown
        if job_title is not None:
            session.job_title = job_title
        if company_name is not None:
            session.company_name = company_name

        db.commit()
        return True, None

    @staticmethod
    def save_session_history(
        db: DBSession,
        session_id: str,
        user_id: str,
        qa_history_json: Optional[str] = None,
        transcript_json: Optional[str] = None,
        summary_markdown: Optional[str] = None,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        session = db.query(AppSession).filter(
            AppSession.id == session_id,
            AppSession.user_id == user_id
        ).first()

        if not session:
            return False, "Session not found"

        if qa_history_json is not None:
            session.qa_history_json = qa_history_json
        if transcript_json is not None:
            session.transcript_json = transcript_json
        if summary_markdown is not None:
            session.summary_markdown = summary_markdown
        if job_title is not None:
            session.job_title = job_title
        if company_name is not None:
            session.company_name = company_name

        db.commit()
        return True, None

    @staticmethod
    def cleanup_expired_leases(db: DBSession) -> int:
        """
        Finds active sessions whose lease expired (e.g. desktop client crashed) and marks them expired.
        """
        now = datetime.utcnow()
        stale_sessions = db.query(AppSession).filter(
            AppSession.status == "active",
            AppSession.lease_expires_at < now
        ).all()

        count = 0
        for s in stale_sessions:
            s.status = "expired"
            s.ended_at = s.lease_expires_at
            count += 1
        
        if count > 0:
            db.commit()
        return count
