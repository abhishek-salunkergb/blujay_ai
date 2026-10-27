import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.config import settings
from backend.database.connection import init_db, SessionLocal, get_db
from backend.services.session_service import SessionService
from backend.api import auth, users, sessions, payments, webhooks, home

# Initialize Database
init_db()

# Background task to clean up expired session leases
async def periodic_lease_cleanup():
    while True:
        db = None
        try:
            db = SessionLocal()
            SessionService.cleanup_expired_leases(db)
        except Exception as e:
            import logging
            logging.getLogger("backend").warning("Lease cleanup notice: %s", e)
        finally:
            if db:
                db.close()
        await asyncio.sleep(45)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(periodic_lease_cleanup())
    try:
        yield
    finally:
        task.cancel()

app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="Authoritative Billing, Session & Credit Backend for Desktop AI Copilot",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(payments.router)
app.include_router(webhooks.router)
app.include_router(home.router)

# Resolve and mount frontend static assets for web dashboard and copilot preview
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def dashboard_view():
    home_path = os.path.join(frontend_dir, "home.html")
    if os.path.isfile(home_path):
        return FileResponse(home_path)
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "billing_model": "30-minute block (50 credit units = 0.5 Credit)",
        "unlimited_subscription": "Available"
    }


@app.get("/copilot", include_in_schema=False)
def copilot_view():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Copilot UI not found")


# Root-level asset fallbacks for relative links
@app.get("/home.css", include_in_schema=False)
def get_home_css():
    return FileResponse(os.path.join(frontend_dir, "home.css"))


@app.get("/home.js", include_in_schema=False)
def get_home_js():
    return FileResponse(os.path.join(frontend_dir, "home.js"))


@app.get("/style.css", include_in_schema=False)
def get_style_css():
    return FileResponse(os.path.join(frontend_dir, "style.css"))


@app.get("/app.js", include_in_schema=False)
def get_app_js():
    return FileResponse(os.path.join(frontend_dir, "app.js"))


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Authoritative production health check probing database connectivity."""
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "degraded", "database": db_status}
        )
    return {
        "status": "healthy",
        "database": db_status,
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "environment": settings.ENV
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
