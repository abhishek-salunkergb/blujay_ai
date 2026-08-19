import os
import json
import time
import threading
try:
    import requests
except ImportError:
    requests = None
from typing import Optional, Callable, Dict, Any, Tuple
from datetime import datetime

# Default backend URL (override via environment or constructor)
DEFAULT_SERVER_URL = os.getenv("COPILOT_BACKEND_URL", "http://127.0.0.1:8000")
TOKEN_STORAGE_PATH = os.path.expanduser("~/.ai_copilot_token.json")


class SessionClient:
    """
    Desktop Client for FastAPI Backend Authentication & Session Billing Management.
    Enforces server-authoritative credit usage, 30-min block leases, and background heartbeats.
    """

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self.access_token: Optional[str] = None
        self.user_data: Optional[Dict[str, Any]] = None
        
        # Active session state
        self.active_session_id: Optional[str] = None
        self.session_expires_at: Optional[float] = None  # Unix timestamp
        self.is_unlimited: bool = False
        self.credit_units: int = 100
        
        # State machine: "IDLE", "STARTING", "ACTIVE", "EXTENDING", "ENDING", "COMPLETED"
        self.state: str = "IDLE"
        
        # Background worker
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat_event = threading.Event()
        
        # Callbacks
        self.on_credits_updated: Optional[Callable[[int, float, bool], None]] = None
        self.on_session_started: Optional[Callable[[str, float], None]] = None
        self.on_session_extended: Optional[Callable[[float, int], None]] = None
        self.on_session_ended: Optional[Callable[[], None]] = None
        self.on_insufficient_credits: Optional[Callable[[str], None]] = None
        self.on_heartbeat_error: Optional[Callable[[str], None]] = None

        # Load token from storage on startup
        self._load_stored_token()

    # =========================================================================
    # TOKEN PERSISTENCE
    # =========================================================================

    def _load_stored_token(self):
        try:
            if os.path.exists(TOKEN_STORAGE_PATH):
                with open(TOKEN_STORAGE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                    if self.access_token:
                        self.fetch_user_profile()
        except Exception:
            self.access_token = None

    def _save_token(self, token: str):
        try:
            self.access_token = token
            with open(TOKEN_STORAGE_PATH, "w", encoding="utf-8") as f:
                json.dump({"access_token": token, "saved_at": datetime.utcnow().isoformat()}, f)
        except Exception:
            pass

    def logout(self):
        self.access_token = None
        self.user_data = None
        self.active_session_id = None
        self.state = "IDLE"
        try:
            if os.path.exists(TOKEN_STORAGE_PATH):
                os.remove(TOKEN_STORAGE_PATH)
        except Exception:
            pass

    def is_authenticated(self) -> bool:
        return bool(self.access_token and self.user_data)

    # =========================================================================
    # HTTP REQUEST HELPER
    # =========================================================================

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================

    def register(self, email: str, password: str) -> Tuple[bool, str]:
        try:
            resp = requests.post(
                f"{self.server_url}/api/auth/register",
                json={"email": email, "password": password},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                self._save_token(data["access_token"])
                self.user_data = data["user"]
                self.credit_units = self.user_data["credit_units"]
                self.is_unlimited = self.user_data["is_unlimited"]
                self._notify_credits()
                return True, "Registration successful!"
            else:
                err = resp.json().get("detail", "Registration failed.")
                return False, err
        except Exception as e:
            return False, f"Server connection error: {e}"

    def login(self, email: str, password: str) -> Tuple[bool, str]:
        try:
            resp = requests.post(
                f"{self.server_url}/api/auth/login",
                json={"email": email, "password": password},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                self._save_token(data["access_token"])
                self.user_data = data["user"]
                self.credit_units = self.user_data["credit_units"]
                self.is_unlimited = self.user_data["is_unlimited"]
                self._notify_credits()
                return True, "Login successful!"
            else:
                err = resp.json().get("detail", "Login failed.")
                return False, err
        except Exception as e:
            return False, f"Server connection error: {e}"

    def fetch_user_profile(self) -> bool:
        if not self.access_token:
            return False
        try:
            resp = requests.get(
                f"{self.server_url}/api/auth/me",
                headers=self._headers(),
                timeout=4
            )
            if resp.status_code == 200:
                self.user_data = resp.json()
                self.credit_units = self.user_data["credit_units"]
                self.is_unlimited = self.user_data["is_unlimited"]
                self._notify_credits()
                return True
            else:
                self.logout()
                return False
        except Exception:
            return False

    def _notify_credits(self):
        if self.on_credits_updated:
            try:
                self.on_credits_updated(
                    self.credit_units,
                    round(self.credit_units / 100.0, 2),
                    self.is_unlimited
                )
            except Exception:
                pass

    # =========================================================================
    # SESSION LIFECYCLE MANAGEMENT
    # =========================================================================

    def start_session(self, mode: str = "copilot") -> Tuple[bool, str]:
        """
        Requests server to start a 30-minute block session.
        Atomically deducts 50 credit units on the server unless user has an active unlimited pass.
        """
        if not self.is_authenticated():
            # Try fetching profile or return auth required
            if not self.fetch_user_profile():
                return False, "Authentication required. Please sign in or create an account."

        self.state = "STARTING"
        try:
            resp = requests.post(
                f"{self.server_url}/api/session/start",
                json={"session_mode": mode},
                headers=self._headers(),
                timeout=6
            )
            if resp.status_code == 200:
                data = resp.json()
                self.active_session_id = data["session_id"]
                self.credit_units = data["remaining_credit_units"]
                self.is_unlimited = data["is_unlimited"]
                
                # Parse ISO timestamp to local unix timestamp
                exp_dt = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                self.session_expires_at = exp_dt.timestamp()
                
                self.state = "ACTIVE"
                self._notify_credits()
                
                # Start background heartbeat thread
                self._start_heartbeat_loop()

                if self.on_session_started:
                    self.on_session_started(self.active_session_id, self.session_expires_at)

                return True, "Session started successfully."
            elif resp.status_code == 402:
                err = resp.json().get("detail", "Insufficient credits (50 units required for 30m block).")
                self.state = "IDLE"
                if self.on_insufficient_credits:
                    self.on_insufficient_credits(err)
                return False, err
            else:
                err = resp.json().get("detail", "Failed to start session.")
                self.state = "IDLE"
                return False, err
        except Exception as e:
            self.state = "IDLE"
            return False, f"Backend connection error: {e}"

    def extend_session(self) -> Tuple[bool, str]:
        """
        Extends the active session by another 30 minutes (-50 units on server).
        """
        if not self.active_session_id:
            return False, "No active session to extend."

        self.state = "EXTENDING"
        try:
            resp = requests.post(
                f"{self.server_url}/api/session/extend",
                json={"session_id": self.active_session_id},
                headers=self._headers(),
                timeout=6
            )
            if resp.status_code == 200:
                data = resp.json()
                self.credit_units = data["remaining_credit_units"]
                self.is_unlimited = data["is_unlimited"]
                
                exp_dt = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                self.session_expires_at = exp_dt.timestamp()
                
                self.state = "ACTIVE"
                self._notify_credits()

                if self.on_session_extended:
                    self.on_session_extended(self.session_expires_at, self.credit_units)

                return True, "Session extended by +30 minutes."
            elif resp.status_code == 402:
                err = resp.json().get("detail", "Insufficient credits to extend session.")
                self.state = "ACTIVE"  # Keep active until time runs out
                if self.on_insufficient_credits:
                    self.on_insufficient_credits(err)
                return False, err
            else:
                err = resp.json().get("detail", "Extension failed.")
                self.state = "ACTIVE"
                return False, err
        except Exception as e:
            self.state = "ACTIVE"
            return False, f"Extension network error: {e}"

    def end_session(
        self,
        qa_pairs: Optional[list] = None,
        transcript_log: Optional[list] = None,
        summary_markdown: Optional[str] = None,
        job_title: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> bool:
        """
        Explicitly ends the session on the backend server and saves history.
        """
        self._stop_heartbeat_event.set()
        if not self.active_session_id:
            self.state = "IDLE"
            return True

        self.state = "ENDING"
        try:
            payload = {
                "session_id": self.active_session_id,
                "qa_history_json": json.dumps(qa_pairs) if qa_pairs else None,
                "transcript_json": json.dumps(transcript_log) if transcript_log else None,
                "summary_markdown": summary_markdown,
                "job_title": job_title,
                "company_name": company_name
            }
            requests.post(
                f"{self.server_url}/api/session/end",
                json=payload,
                headers=self._headers(),
                timeout=5
            )
        except Exception:
            pass

        self.active_session_id = None
        self.session_expires_at = None
        self.state = "COMPLETED"
        
        if self.on_session_ended:
            self.on_session_ended()
            
        self.state = "IDLE"
        return True

    def get_history(self, limit: int = 20) -> list:
        if not self.is_authenticated():
            return []
        try:
            resp = requests.get(
                f"{self.server_url}/api/session/history?limit={limit}",
                headers=self._headers(),
                timeout=5
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    # =========================================================================
    # HEARTBEAT & AUTO-EXTENSION LOOP
    # =========================================================================

    def _start_heartbeat_loop(self):
        self._stop_heartbeat_event.clear()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            daemon=True,
            name="SessionClientHeartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_worker(self):
        """
        Runs quietly in background:
         1. Sends heartbeat to server every 25 seconds (keeps 2-minute lease refreshed).
         2. When 45 seconds remain before 30-min block expiration, triggers extend_session().
        """
        while not self._stop_heartbeat_event.is_set():
            time.sleep(15)
            if not self.active_session_id or self.state not in ["ACTIVE", "EXTENDING"]:
                break

            # 1. Send Heartbeat
            try:
                resp = requests.post(
                    f"{self.server_url}/api/session/heartbeat",
                    json={"session_id": self.active_session_id},
                    headers=self._headers(),
                    timeout=4
                )
                if resp.status_code == 200:
                    hb_data = resp.json()
                    if not hb_data.get("success"):
                        # Session expired or invalidated on server
                        if self.on_heartbeat_error:
                            self.on_heartbeat_error(hb_data.get("message", "Session lease expired"))
                        self.end_session()
                        break
            except Exception as e:
                # Network hiccup, will retry in next loop
                pass

            # 2. Check if auto-extension is needed (< 45s remaining)
            if self.session_expires_at:
                remaining_sec = self.session_expires_at - time.time()
                if 0 < remaining_sec <= 45 and self.state == "ACTIVE":
                    # Attempt extension
                    self.extend_session()

    def get_remaining_seconds(self) -> int:
        if not self.session_expires_at:
            return 0
        return max(0, int(self.session_expires_at - time.time()))

    # =========================================================================
    # PAYMENTS & PRODUCTS
    # =========================================================================

    def get_products(self) -> list:
        try:
            resp = requests.get(f"{self.server_url}/api/payments/products", timeout=4)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def mock_buy_product(self, product_id: str) -> Tuple[bool, str]:
        """
        Instant local test purchase without real card.
        """
        try:
            resp = requests.post(
                f"{self.server_url}/api/payments/mock-checkout",
                json={"product_id": product_id},
                headers=self._headers(),
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                self.credit_units = data["credit_units"]
                self.is_unlimited = data["is_unlimited"]
                self._notify_credits()
                return True, data.get("message", "Credits added successfully!")
            else:
                return False, resp.json().get("detail", "Payment failed.")
        except Exception as e:
            return False, f"Payment error: {e}"


# Global Singleton Client Instance
client = SessionClient()
