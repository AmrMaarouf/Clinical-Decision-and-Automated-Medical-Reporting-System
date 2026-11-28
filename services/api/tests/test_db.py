from fastapi.testclient import TestClient

from src.db import Artifact, Case, SessionLocal, Task
from src.main import app

client = TestClient(app)


def test_everything_is_saved_in_the_database(monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_S", "0")
    case_id = client.post("/cases").json()["case_id"]
    task_id = client.post(f"/cases/{case_id}/process").json()["task_id"]

    # read straight from the database, not through the API
    with SessionLocal() as db:
        assert db.get(Case, case_id) is not None
        assert db.get(Task, task_id).status == "done"
        assert db.get(Artifact, (case_id, "report")).data["case_id"] == case_id
        for stage in ("asr", "extract", "labs", "xray", "report"):
            assert db.get(Artifact, (case_id, stage)) is not None


def test_processing_again_replaces_old_outputs(monkeypatch):
    monkeypatch.setenv("FAKE_DELAY_S", "0")
    case_id = client.post("/cases").json()["case_id"]
    client.post(f"/cases/{case_id}/process")

    monkeypatch.setenv("LABS_FAKE", "0")        # second run: labs fails
    client.post(f"/cases/{case_id}/process")

    with SessionLocal() as db:
        assert db.get(Artifact, (case_id, "labs")) is None   # old labs output was removed
        assert db.get(Artifact, (case_id, "report")) is not None
