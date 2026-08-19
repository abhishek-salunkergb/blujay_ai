import json
import asyncio
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from backend.database.connection import get_db, SessionLocal
from backend.database.models import User, Session as DBSession
from backend.services.auth_service import get_current_user, decode_access_token
from backend.services.session_service import SessionService
import os
from groq import Groq

router = APIRouter(prefix="/api/session", tags=["Sessions"])


class StartSessionRequest(BaseModel):
    session_mode: str = "copilot"


class SessionIdRequest(BaseModel):
    session_id: str


class EndSessionRequest(BaseModel):
    session_id: str
    qa_history_json: str | None = None
    transcript_json: str | None = None
    summary_markdown: str | None = None
    job_title: str | None = None
    company_name: str | None = None


class SaveHistoryRequest(BaseModel):
    session_id: str
    qa_history_json: str | None = None
    transcript_json: str | None = None
    summary_markdown: str | None = None
    job_title: str | None = None
    company_name: str | None = None


class SessionResponse(BaseModel):
    success: bool
    session_id: str | None
    status: str
    expires_at: str | None
    lease_expires_at: str | None
    credit_units_used: int
    remaining_credit_units: int
    remaining_credits_formatted: float
    is_unlimited: bool
    message: str | None = None


class HeartbeatResponse(BaseModel):
    success: bool
    status: str
    expires_at: str | None
    lease_expires_at: str | None
    message: str | None = None


class SessionHistorySummary(BaseModel):
    id: str
    session_mode: str
    started_at: str
    ended_at: str | None
    status: str
    job_title: str | None
    company_name: str | None
    summary_markdown: str | None
    total_questions: int = 0


@router.post("/start", response_model=SessionResponse)
def start_session(
    req: StartSessionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    success, session, error_msg, remaining_units = SessionService.start_session(
        db=db,
        user=user,
        mode=req.session_mode
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=error_msg or "Insufficient credits to start session."
        )

    return SessionResponse(
        success=True,
        session_id=session.id,
        status=session.status,
        expires_at=session.expires_at.isoformat(),
        lease_expires_at=session.lease_expires_at.isoformat(),
        credit_units_used=session.credit_units_used,
        remaining_credit_units=remaining_units,
        remaining_credits_formatted=round(remaining_units / 100.0, 2),
        is_unlimited=user.has_unlimited_subscription(),
        message="Session started successfully (30-minute block authorized)"
    )


@router.post("/heartbeat", response_model=HeartbeatResponse)
def heartbeat(
    req: SessionIdRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    success, error_msg, expires_at, lease_expires_at = SessionService.heartbeat(
        db=db,
        session_id=req.session_id,
        user_id=user.id
    )

    if not success:
        return HeartbeatResponse(
            success=False,
            status="expired" if "expired" in (error_msg or "") else "error",
            expires_at=expires_at.isoformat() if expires_at else None,
            lease_expires_at=lease_expires_at.isoformat() if lease_expires_at else None,
            message=error_msg
        )

    return HeartbeatResponse(
        success=True,
        status="active",
        expires_at=expires_at.isoformat() if expires_at else None,
        lease_expires_at=lease_expires_at.isoformat() if lease_expires_at else None,
        message="Heartbeat acknowledged"
    )


@router.post("/extend", response_model=SessionResponse)
def extend_session(
    req: SessionIdRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    success, error_msg, remaining_units, new_expires_at = SessionService.extend_session(
        db=db,
        session_id=req.session_id,
        user=user
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=error_msg or "Insufficient credits to extend session."
        )

    return SessionResponse(
        success=True,
        session_id=req.session_id,
        status="active",
        expires_at=new_expires_at.isoformat() if new_expires_at else None,
        lease_expires_at=None,
        credit_units_used=50,
        remaining_credit_units=remaining_units,
        remaining_credits_formatted=round(remaining_units / 100.0, 2),
        is_unlimited=user.has_unlimited_subscription(),
        message="Session successfully extended by +30 minutes"
    )


@router.post("/end")
def end_session(
    req: EndSessionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    success, error_msg = SessionService.end_session(
        db=db,
        session_id=req.session_id,
        user_id=user.id,
        qa_history_json=req.qa_history_json,
        transcript_json=req.transcript_json,
        summary_markdown=req.summary_markdown,
        job_title=req.job_title,
        company_name=req.company_name
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)

    return {"success": True, "message": "Session completed and history saved."}


@router.post("/save-history")
def save_session_history(
    req: SaveHistoryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    success, error_msg = SessionService.save_session_history(
        db=db,
        session_id=req.session_id,
        user_id=user.id,
        qa_history_json=req.qa_history_json,
        transcript_json=req.transcript_json,
        summary_markdown=req.summary_markdown,
        job_title=req.job_title,
        company_name=req.company_name
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)

    return {"success": True, "message": "Session history saved."}


@router.get("/history", response_model=list[SessionHistorySummary])
def get_user_session_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20
):
    sessions = db.query(DBSession).filter(
        DBSession.user_id == user.id
    ).order_by(DBSession.created_at.desc()).limit(limit).all()

    results = []
    for s in sessions:
        total_q = 0
        if s.qa_history_json:
            try:
                qa_data = json.loads(s.qa_history_json)
                total_q = len(qa_data) if isinstance(qa_data, list) else 0
            except Exception:
                pass

        results.append(SessionHistorySummary(
            id=s.id,
            session_mode=s.session_mode,
            started_at=s.started_at.isoformat() if s.started_at else "",
            ended_at=s.ended_at.isoformat() if s.ended_at else None,
            status=s.status,
            job_title=s.job_title,
            company_name=s.company_name,
            summary_markdown=s.summary_markdown,
            total_questions=total_q
        ))
    return results


@router.get("/{session_id}")
def get_session_details(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    s = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == user.id
    ).first()

    if not s:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    qa_list = []
    if s.qa_history_json:
        try:
            qa_list = json.loads(s.qa_history_json)
        except Exception:
            pass

    transcript_list = []
    if s.transcript_json:
        try:
            transcript_list = json.loads(s.transcript_json)
        except Exception:
            pass

    return {
        "id": s.id,
        "session_mode": s.session_mode,
        "started_at": s.started_at.isoformat() if s.started_at else "",
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "status": s.status,
        "job_title": s.job_title,
        "company_name": s.company_name,
        "summary_markdown": s.summary_markdown,
        "qa_pairs": qa_list,
        "transcript_log": transcript_list,
        "credit_units_used": s.credit_units_used
    }


# ============================================================
# HIGH-PERFORMANCE WEBSOCKET STREAMING ENDPOINT
# ============================================================

@router.websocket("/ws/{session_id}")
async def websocket_session_stream(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(None)
):
    """
    Real-time duplex WebSocket stream for high-concurrency interview copilot clients.
    Transfers questions, streams tokens word-by-word with <300ms latency.
    """
    await websocket.accept()
    db = SessionLocal()

    try:
        # 1. Authenticate user from query token if provided
        user = None
        if token:
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                user = db.query(User).filter(User.id == payload["sub"]).first()

        # 2. Verify Session Validity
        session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not session or session.status != "active":
            await websocket.send_json({"type": "error", "message": "Invalid or expired session"})
            await websocket.close()
            return

        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "status": "ready",
            "message": "WebSocket streaming pipeline connected"
        })

        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

            elif action == "ask_stream":
                question = payload.get("question", "")
                context = payload.get("context", "")
                
                if not groq_client:
                    await websocket.send_json({"type": "token", "token": "Error: Groq client not configured on server."})
                    await websocket.send_json({"type": "done", "full_text": "Error: Groq client not configured on server."})
                    continue

                prompt = (
                    "You are a real-time interview copilot. Provide direct, concise, high-impact bullet points. "
                    "Structure technical concepts clearly. If code is needed, provide only clean idiomatic code.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )

                try:
                    response_stream = groq_client.chat.completions.create(
                        model="qwen/qwen3.6-27b",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        temperature=0.2,
                        max_tokens=500
                    )

                    full_response = ""
                    for chunk in response_stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else ""
                        if delta:
                            full_response += delta
                            await websocket.send_json({"type": "token", "token": delta})
                            await asyncio.sleep(0.001)

                    await websocket.send_json({"type": "done", "full_text": full_response})

                except Exception as stream_err:
                    await websocket.send_json({"type": "error", "message": str(stream_err)})

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket session error: {e}")
    finally:
        db.close()
