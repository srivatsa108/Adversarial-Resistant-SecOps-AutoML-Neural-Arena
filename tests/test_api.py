"""Integration tests for FastAPI backend endpoints."""

import io
from fastapi.testclient import TestClient
from app_fastapi import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["zero_trust_privacy"] == "active"


def test_ingest_and_clean_endpoint():
    """Verify in-memory ingestion and cleaning endpoints."""
    csv_content = b"host,port,status\n192.168.1.15,80,OK\n192.168.1.15,80,OK\n10.0.0.4,443,OK\n"
    files = {"file": ("network.csv", io.BytesIO(csv_content), "text/csv")}
    response = client.post(
        "/api/v1/ingest",
        files=files,
        data={"enable_zero_trust_masking": "true", "session_id": "test_session"},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["detected_format"] == "csv"

    # Clean endpoint
    clean_res = client.post(
        "/api/v1/clean",
        data={"session_id": "test_session"},
    )
    assert clean_res.status_code == 200
    clean_data = clean_res.json()
    assert clean_data["status"] == "success"
    assert "cleaning_report" in clean_data
