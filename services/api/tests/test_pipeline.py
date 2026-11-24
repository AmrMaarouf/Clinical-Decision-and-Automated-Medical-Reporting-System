import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_S", "0")


def _process_new_case():
    case_id = client.post("/cases").json()["case_id"]
    task_id = client.post(f"/cases/{case_id}/process").json()["task_id"]
    # TestClient finishes background tasks before returning the response
    return case_id, client.get(f"/tasks/{task_id}").json()


def test_full_fake_pipeline():
    case_id, task = _process_new_case()
    assert task["status"] == "done"
    assert set(task["stages"].values()) == {"done"}

    report = client.get(f"/cases/{case_id}/report").json()
    assert report["case_id"] == case_id
    assert report["status"] == "draft"
    assert len(report["sections"]) == 7


def test_artifact_endpoint_returns_transcript():
    case_id, _ = _process_new_case()
    r = client.get(f"/cases/{case_id}/artifacts/asr")
    assert r.status_code == 200
    assert r.json()["case_id"] == case_id


def test_partial_when_labs_fail(monkeypatch):
    monkeypatch.setenv("LABS_FAKE", "0")     # real labs service is not connected -> fails
    case_id, task = _process_new_case()
    assert task["stages"]["labs"] == "failed"
    assert "labs" in task["errors"]
    assert task["status"] == "partial"
    assert client.get(f"/cases/{case_id}/report").status_code == 200


def test_failed_asr_skips_dependent_stages(monkeypatch):
    monkeypatch.setenv("ASR_FAKE", "0")
    _, task = _process_new_case()
    assert task["stages"]["asr"] == "failed"
    assert task["stages"]["extract"] == "skipped"
    assert task["stages"]["report"] == "skipped"
    assert task["status"] == "failed"


def test_report_before_processing_is_409():
    case_id = client.post("/cases").json()["case_id"]
    assert client.get(f"/cases/{case_id}/report").status_code == 409


def test_unknown_ids_are_404():
    assert client.post("/cases/nope/process").status_code == 404
    assert client.get("/tasks/nope").status_code == 404
    assert client.get("/cases/nope/report").status_code == 404
