import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from sqlalchemy import delete

from .db import Artifact, Case, SessionLocal, Task, init_db
from .fakes import STAGES, run_stage

SERVICE = "api"
app = FastAPI(title=f"mica-{SERVICE}")

# stage -> stages that must be "done" before it can run
DEPENDS_ON = {"extract": ["asr"], "report": ["extract"]}

init_db()   # create the tables on startup if they are missing


@app.get("/health")
def health():
    return {
        "service": SERVICE,
        "status": "ok",
        "fake": os.getenv("API_FAKE", "1") == "1",
    }


def _case_or_404(db, case_id: str) -> None:
    if db.get(Case, case_id) is None:
        raise HTTPException(status_code=404, detail="case not found")


def _task_dict(task: Task) -> dict:
    return {
        "task_id": task.id,
        "case_id": task.case_id,
        "status": task.status,
        "stages": task.stages,
        "errors": task.errors,
    }


@app.post("/cases", status_code=201)
def create_case():
    case_id = "c" + uuid.uuid4().hex[:8]
    with SessionLocal() as db:
        db.add(Case(id=case_id))
        db.commit()
    return {"case_id": case_id}


def _set_stage(db, task: Task, stage: str, state: str, error: str | None = None) -> None:
    # always assign a NEW dict: SQLAlchemy does not notice changes made inside a JSON dict
    task.stages = {**task.stages, stage: state}
    if error:
        task.errors = {**task.errors, stage: error}
    db.commit()


def run_pipeline(task_id: str) -> None:
    """Runs all stages in order. Called in the background (later: by a Celery worker)."""
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        case_id = task.case_id
        task.status = "running"
        db.commit()

        for stage in STAGES:
            if any(task.stages[dep] != "done" for dep in DEPENDS_ON.get(stage, [])):
                _set_stage(db, task, stage, "skipped")
                continue
            _set_stage(db, task, stage, "running")
            try:
                data = run_stage(stage, case_id)
                db.merge(Artifact(case_id=case_id, stage=stage, data=data))
                _set_stage(db, task, stage, "done")      # saves output + state together
            except Exception as exc:                     # a failed stage must not kill the others
                db.rollback()
                _set_stage(db, task, stage, "failed", f"{type(exc).__name__}: {exc}")

        states = task.stages
        if all(s == "done" for s in states.values()):
            task.status = "done"
        elif states["report"] == "done":
            task.status = "partial"      # report exists but some stage failed
        else:
            task.status = "failed"
        db.commit()


@app.post("/cases/{case_id}/process", status_code=202)
def process_case(case_id: str, background_tasks: BackgroundTasks):
    task_id = "t" + uuid.uuid4().hex[:8]
    with SessionLocal() as db:
        _case_or_404(db, case_id)
        # start clean if the case is processed again
        db.execute(delete(Artifact).where(Artifact.case_id == case_id))
        db.add(Task(
            id=task_id,
            case_id=case_id,
            status="queued",
            stages={stage: "pending" for stage in STAGES},
            errors={},
        ))
        db.commit()
    background_tasks.add_task(run_pipeline, task_id)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _task_dict(task)


@app.get("/cases/{case_id}/report")
def get_report(case_id: str):
    with SessionLocal() as db:
        _case_or_404(db, case_id)
        artifact = db.get(Artifact, (case_id, "report"))
        if artifact is None:
            raise HTTPException(
                status_code=409, detail="report not ready: call /process and wait for the task"
            )
        return artifact.data


@app.get("/cases/{case_id}/artifacts/{stage}")
def get_artifact(case_id: str, stage: str):
    """Raw output of one stage (transcript, extraction, ...). The UI needs the
    transcript to show the evidence behind each report sentence."""
    if stage not in STAGES:
        raise HTTPException(status_code=404, detail=f"unknown stage, use one of {STAGES}")
    with SessionLocal() as db:
        _case_or_404(db, case_id)
        artifact = db.get(Artifact, (case_id, stage))
        if artifact is None:
            raise HTTPException(status_code=409, detail=f"{stage} output not available")
        return artifact.data
