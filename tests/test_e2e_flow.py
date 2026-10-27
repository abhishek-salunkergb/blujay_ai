import sys
import os
sys.path.insert(0, os.path.abspath("."))
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_full_system_e2e_flow():
    print("Testing Full System E2E Flow...")

    import uuid
    unique_email = f"sarah_e2e_{uuid.uuid4().hex[:6]}@ai-copilot.io"
    reg_resp = client.post("/api/auth/register", json={
        "email": unique_email,
        "password": "SecurePassword123"
    })
    assert reg_resp.status_code == 200, reg_resp.text
    reg_data = reg_resp.json()
    token = reg_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert reg_data["user"]["credit_units"] == 100
    assert reg_data["user"]["sessions_remaining"] == 2

    # 2. Check Credit Balance API
    bal_resp = client.get("/api/user/credits", headers=headers)
    assert bal_resp.status_code == 200
    assert bal_resp.json()["credit_units"] == 100

    # 3. Start Session -> Deducts 50 units (0.5 Credit)
    start_resp = client.post("/api/session/start", json={"session_mode": "copilot"}, headers=headers)
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert start_data["remaining_credit_units"] == 50

    # 4. Send Heartbeat -> Acknowledged
    hb_resp = client.post("/api/session/heartbeat", json={"session_id": session_id}, headers=headers)
    assert hb_resp.status_code == 200
    assert hb_resp.json()["success"] is True

    # 5. Extend Session -> Deducts another 50 units (Remaining: 0)
    ext_resp = client.post("/api/session/extend", json={"session_id": session_id}, headers=headers)
    assert ext_resp.status_code == 200
    assert ext_resp.json()["remaining_credit_units"] == 0

    # 6. Attempt 2nd Extension -> Fails with 402 Insufficient credits
    ext_fail = client.post("/api/session/extend", json={"session_id": session_id}, headers=headers)
    assert ext_fail.status_code == 402
    assert "Insufficient credits" in ext_fail.json()["detail"]

    # 7. End Active Session
    end_resp = client.post("/api/session/end", json={"session_id": session_id}, headers=headers)
    assert end_resp.status_code == 200

    # 8. User purchases Pro Pack (+500 units)
    buy_resp = client.post("/api/payments/mock-checkout", json={"product_id": "pro_pack"}, headers=headers)
    assert buy_resp.status_code == 200
    assert buy_resp.json()["credit_units"] == 500

    # 9. Start New Session -> Deducts 50 units (Remaining: 450)
    s2_resp = client.post("/api/session/start", json={"session_mode": "copilot"}, headers=headers)
    assert s2_resp.status_code == 200
    assert s2_resp.json()["remaining_credit_units"] == 450

    # 10. Check Transaction Ledger
    tx_resp = client.get("/api/user/transactions", headers=headers)
    assert tx_resp.status_code == 200
    txs = tx_resp.json()
    assert len(txs) >= 4  # session_start, session_extension, purchase, session_start

    print("SUCCESS: FULL SYSTEM E2E TEST PASSED 100%!")
    client.close()


if __name__ == "__main__":
    test_full_system_e2e_flow()
