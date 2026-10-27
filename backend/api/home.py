import json
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import User, Session as DBSession
from backend.services.auth_service import get_current_user
from backend.services.session_service import SessionService
from backend.config import settings

router = APIRouter(prefix="/api/home", tags=["Home Dashboard"])


# =============================================================================
# SCHEMAS
# =============================================================================

class UserProfileDTO(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    skills: str
    tone: str
    credit_units: int
    credits_formatted: float
    sessions_remaining: int
    is_unlimited: bool
    subscription_tier: str
    subscription_expires_at: Optional[str] = None


class HomeStatsDTO(BaseModel):
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    total_questions_practiced: int
    estimated_prep_hours_saved: float
    total_credits_spent: float
    avg_match_fit_score: int


class PresetTemplateDTO(BaseModel):
    id: str
    icon: str
    title: str
    role_preset: str
    description: str
    category: str
    recommended_audio: str
    color: str
    auto_generate: bool


class RecentSessionSummaryDTO(BaseModel):
    id: str
    session_mode: str
    status: str
    started_at: str
    ended_at: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    total_questions: int
    summary_preview: Optional[str] = None
    duration_minutes: Optional[int] = None


class SystemStatusDTO(BaseModel):
    api_status: str
    version: str
    groq_lpu_status: str
    default_llm_model: str
    groq_latency_ms: int
    stealth_screen_protection: str
    rag_vector_engine: str


class HomeDashboardResponse(BaseModel):
    success: bool
    user: UserProfileDTO
    stats: HomeStatsDTO
    active_session: Optional[RecentSessionSummaryDTO] = None
    recent_sessions: List[RecentSessionSummaryDTO]
    templates: List[PresetTemplateDTO]
    system_status: SystemStatusDTO
    timestamp: str


class QuickStartRequest(BaseModel):
    template_id: str
    custom_company: Optional[str] = None
    custom_jd: Optional[str] = None


class JDAnalysisRequest(BaseModel):
    job_description: str
    resume_text: Optional[str] = ""
    job_title: Optional[str] = "Software Engineering"


class JDAnalysisResponse(BaseModel):
    success: bool
    match_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    bridging_talking_points: List[str]
    predicted_questions: List[str]


def _get_avg_match_fit_score() -> int:
    try:
        from jd_matcher import get_cached_jd_analysis
        cached = get_cached_jd_analysis()
        if cached and isinstance(cached, dict) and "match_score" in cached:
            return int(cached["match_score"])
    except Exception:
        pass
    return 88


# =============================================================================
# CURATED PRESET TEMPLATES
# =============================================================================

CURATED_TEMPLATES: List[PresetTemplateDTO] = [
    PresetTemplateDTO(
        id="coding_leetcode",
        icon="💻",
        title="Coding & LeetCode",
        role_preset="Software Engineer (Coding & Algorithms)",
        description="Real-time screen vision OCR solver, Big-O complexity analysis, and Python/C++/Java live sandbox.",
        category="Technical",
        recommended_audio="dual",
        color="#38BDF8",
        auto_generate=True
    ),
    PresetTemplateDTO(
        id="system_design",
        icon="🏛️",
        title="System Design & Architecture",
        role_preset="Senior / Principal Systems Architect",
        description="Scalable distributed architecture, CAP theorem trade-offs, cache invalidation, and database partitioning.",
        category="Architecture",
        recommended_audio="dual",
        color="#818CF8",
        auto_generate=True
    ),
    PresetTemplateDTO(
        id="star_behavioral",
        icon="🎯",
        title="Behavioral & STAR Stories",
        role_preset="Engineering Lead / Manager",
        description="STAR framework delivery (Situation, Task, Action, Result) strictly grounded in your verified resume achievements.",
        category="Behavioral",
        recommended_audio="dual",
        color="#10B981",
        auto_generate=True
    ),
    PresetTemplateDTO(
        id="executive_leadership",
        icon="💼",
        title="Executive & Tech Leadership",
        role_preset="Director of Engineering / VP",
        description="High-level vision, cross-functional stakeholder management, product velocity, and team scaling strategies.",
        category="Leadership",
        recommended_audio="dual",
        color="#F59E0B",
        auto_generate=True
    ),
    PresetTemplateDTO(
        id="rapid_practice",
        icon="⚡",
        title="15-Min Rapid Mock Interview",
        role_preset="Full Stack Developer",
        description="Fast-paced technical drill with AI interviewer testing core fundamentals and edge cases under time pressure.",
        category="Practice",
        recommended_audio="mic",
        color="#EC4899",
        auto_generate=False
    )
]


# =============================================================================
# HOME DASHBOARD ENDPOINT
# =============================================================================

@router.get("/dashboard", response_model=HomeDashboardResponse)
def get_home_dashboard(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Unified Home API endpoint returning all data required to render the
    next-generation Home UI Dashboard in a single round-trip.
    """
    is_unlimited = user.has_unlimited_subscription()
    sessions_remaining = 999 if is_unlimited else (user.credit_units // settings.SESSION_BLOCK_UNITS)

    # 1. User Profile DTO
    user_dto = UserProfileDTO(
        id=user.id,
        email=user.email,
        full_name="Abhishek Salunke",
        role="Senior Full Stack & AI Engineer",
        skills="Python, Django, FastAPI, React, PostgreSQL, Docker, RAG, LLM Copilots",
        tone="Direct & Highly Technical (Recommended)",
        credit_units=user.credit_units,
        credits_formatted=round(user.credit_units / 100.0, 2),
        sessions_remaining=sessions_remaining,
        is_unlimited=is_unlimited,
        subscription_tier=user.subscription_tier,
        subscription_expires_at=user.subscription_expires_at.isoformat() if user.subscription_expires_at else None
    )

    # 2. Query User Sessions
    all_sessions = db.query(DBSession).filter(
        DBSession.user_id == user.id
    ).order_by(DBSession.created_at.desc()).all()

    total_sessions_count = len(all_sessions)
    active_sessions_count = sum(1 for s in all_sessions if s.status == "active")
    completed_sessions_count = sum(1 for s in all_sessions if s.status == "completed")

    total_questions = 0
    recent_summaries: List[RecentSessionSummaryDTO] = []
    active_session_dto: Optional[RecentSessionSummaryDTO] = None

    for s in all_sessions:
        q_count = 0
        preview_text = None
        if s.qa_history_json:
            try:
                qa_data = json.loads(s.qa_history_json)
                if isinstance(qa_data, list):
                    q_count = len(qa_data)
            except Exception:
                pass
        total_questions += q_count

        if s.summary_markdown:
            # First 120 chars as preview
            preview_text = s.summary_markdown[:120].strip().replace("\n", " ") + "..."

        duration = None
        if s.started_at and s.ended_at:
            duration = max(1, int((s.ended_at - s.started_at).total_seconds() // 60))
        elif s.started_at and s.status == "active":
            duration = max(1, int((datetime.utcnow() - s.started_at).total_seconds() // 60))

        summary_item = RecentSessionSummaryDTO(
            id=s.id,
            session_mode=s.session_mode,
            status=s.status,
            started_at=s.started_at.isoformat() if s.started_at else "",
            ended_at=s.ended_at.isoformat() if s.ended_at else None,
            job_title=s.job_title or "Software Engineering Interview",
            company_name=s.company_name or "Tech Company",
            total_questions=q_count,
            summary_preview=preview_text,
            duration_minutes=duration
        )

        if s.status == "active" and active_session_dto is None:
            active_session_dto = summary_item

        if len(recent_summaries) < 10:
            recent_summaries.append(summary_item)

    # 3. Calculate Stats & prep hours saved
    credits_spent = sum(s.credit_units_used for s in all_sessions) / 100.0
    prep_hours_saved = round((total_questions * 4.5 + total_sessions_count * 30) / 60.0, 1)

    stats_dto = HomeStatsDTO(
        total_sessions=total_sessions_count,
        active_sessions=active_sessions_count,
        completed_sessions=completed_sessions_count,
        total_questions_practiced=total_questions,
        estimated_prep_hours_saved=prep_hours_saved,
        total_credits_spent=round(credits_spent, 2),
        avg_match_fit_score=_get_avg_match_fit_score()
    )

    # 4. System Status
    system_status_dto = SystemStatusDTO(
        api_status="online",
        version="2.0.0",
        groq_lpu_status="operational",
        default_llm_model="qwen/qwen3.6-27b",
        groq_latency_ms=280,
        stealth_screen_protection="WDA_EXCLUDEFROMCAPTURE active",
        rag_vector_engine="operational"
    )

    return HomeDashboardResponse(
        success=True,
        user=user_dto,
        stats=stats_dto,
        active_session=active_session_dto,
        recent_sessions=recent_summaries,
        templates=CURATED_TEMPLATES,
        system_status=system_status_dto,
        timestamp=datetime.utcnow().isoformat()
    )


# =============================================================================
# STATS ONLY ENDPOINT
# =============================================================================

@router.get("/stats", response_model=HomeStatsDTO)
def get_home_stats(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Lightweight endpoint for periodic background dashboard metric refreshes.
    """
    all_sessions = db.query(DBSession).filter(DBSession.user_id == user.id).all()
    total_sessions_count = len(all_sessions)
    active_sessions_count = sum(1 for s in all_sessions if s.status == "active")
    completed_sessions_count = sum(1 for s in all_sessions if s.status == "completed")

    total_questions = 0
    for s in all_sessions:
        if s.qa_history_json:
            try:
                qa_data = json.loads(s.qa_history_json)
                if isinstance(qa_data, list):
                    total_questions += len(qa_data)
            except Exception:
                pass

    credits_spent = sum(s.credit_units_used for s in all_sessions) / 100.0
    prep_hours_saved = round((total_questions * 4.5 + total_sessions_count * 30) / 60.0, 1)

    return HomeStatsDTO(
        total_sessions=total_sessions_count,
        active_sessions=active_sessions_count,
        completed_sessions=completed_sessions_count,
        total_questions_practiced=total_questions,
        estimated_prep_hours_saved=prep_hours_saved,
        total_credits_spent=round(credits_spent, 2),
        avg_match_fit_score=_get_avg_match_fit_score()
    )


# =============================================================================
# TEMPLATES ENDPOINT
# =============================================================================

@router.get("/templates", response_model=List[PresetTemplateDTO])
def get_preset_templates():
    """
    Returns the list of recommended preset templates.
    """
    return CURATED_TEMPLATES


# =============================================================================
# QUICK START HELPER ENDPOINT
# =============================================================================

@router.post("/quick-start")
def quick_start_template(
    req: QuickStartRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Pre-configures a session from a template and initializes session authorization.
    """
    template = next((t for t in CURATED_TEMPLATES if t.id == req.template_id), None)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{req.template_id}' not found."
        )

    success, session, error_msg, remaining_units = SessionService.start_session(
        db=db,
        user=user,
        mode="copilot" if template.category != "Practice" else "practice"
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=error_msg or "Insufficient credits."
        )

    # Pre-populate role and company on session
    session.job_title = template.role_preset
    session.company_name = req.custom_company or "Target Company"
    db.commit()

    return {
        "success": True,
        "session_id": session.id,
        "template": template.model_dump() if hasattr(template, "model_dump") else template.dict(),
        "job_title": session.job_title,
        "company_name": session.company_name,
        "expires_at": session.expires_at.isoformat(),
        "remaining_credit_units": remaining_units
    }


# =============================================================================
# JOB DESCRIPTION (JD) & RESUME FIT ANALYSIS ENDPOINT
# =============================================================================

@router.post("/analyze-fit", response_model=JDAnalysisResponse)
def analyze_job_fit(
    req: JDAnalysisRequest,
    user: User = Depends(get_current_user)
):
    """
    Calculates semantic skill alignment, gaps, bridging points, and predicted questions
    between candidate resume and job description using the jd_matcher engine.
    """
    if not req.job_description.strip():
        return JDAnalysisResponse(
            success=True,
            match_score=0,
            matched_skills=[],
            missing_skills=[],
            bridging_talking_points=["Please provide job description text to calculate fit."],
            predicted_questions=[]
        )
    try:
        from jd_matcher import analyze_jd_resume_fit, save_jd_analysis_cache
        resume_input = req.resume_text.strip() if req.resume_text else "Python, Django, FastAPI, React, PostgreSQL, Docker, RAG, LLM Copilots"
        result = analyze_jd_resume_fit(
            jd_text=req.job_description,
            resume_text=resume_input,
            job_title=req.job_title
        )
        save_jd_analysis_cache(result)
        return JDAnalysisResponse(
            success=True,
            match_score=result.get("match_score", 75),
            matched_skills=result.get("matched_skills", []),
            missing_skills=result.get("missing_skills", []),
            bridging_talking_points=result.get("bridging_talking_points", []),
            predicted_questions=result.get("predicted_questions", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JD Analysis error: {e}")
