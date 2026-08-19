import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database.connection import init_db, SessionLocal
from backend.services.session_service import SessionService
from backend.api import auth, users, sessions, payments, webhooks

# Initialize Database
init_db()

app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="Authoritative Billing, Session & Credit Backend for Desktop AI Copilot"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/")
def root():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "billing_model": "30-minute block (50 credit units = 0.5 Credit)",
        "unlimited_subscription": "Available"
    }


# Background task to clean up expired session leases
async def periodic_lease_cleanup():
    while True:
        try:
            db = SessionLocal()
            SessionService.cleanup_expired_leases(db)
            db.close()
        except Exception:
            pass
        await asyncio.sleep(45)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_lease_cleanup())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
