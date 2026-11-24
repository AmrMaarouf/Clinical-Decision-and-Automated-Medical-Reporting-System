import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException

from .fakes import STAGES, run_stage

SERVICE = "api"
app = FastAPI(title=f"mica-{SERVICE}")

# stage -> stages that must be "done" before it can run
DEPENDS_ON = {"extract": ["asr"], "report": ["extract"]}

# In-memory storage (step 1 only). Lost on restart. Replaced by Postgres later.
CASES: dict[str, dict] = {}
TASKS: dict[str, dict] = {}
ARTIFACTS: dict[str, dict[str, dict]] = {}   # case_id -> stage -> JSON output


@app.get("/health")
def health():
    return {
        "service": SERVICE,
        "status": "ok",
        "fake": os.getenv("API_FAKE", "1") == "1",
    }


def _case_or_404(case_id: str) -> None:
    if case_id not in CASES:
        raise HTTPException(status_code=404, detail="case not found")


@app.post("/cases", status_code=201)
def create_case():
    case_id = "c" + uuid.uuid4().hex[:8]
    CASES[case_id] = {"case_id": case_id}
    ARTIFACTS[case_id] = {}
    return {"case_id": case_id}


def run_pipeline(task_id: str) -> None:
    """Runs all stages in order. Called in the background (later: by a Celery worker)."""
    task = TASKS[task_id]
    case_id = task["case_id"]
    task["status"] = "running"

    for stage in STAGES:
        if any(task["stages"][dep] != "done" for dep in DEPENDS_ON.get(stage, [])):
            task["stages"][stage] = "skipped"
            continue
        task["stages"][stage] = "running"
        try:
            ARTIFACTS[case_id][stage] = run_stage(stage, case_id)
            task["stages"][stage] = "done"
        except Exception as exc:  # a failed stage must not kill the others
            task["stages"][stage] = "failed"
            task["errors"][stage] = f"{type(exc).__name__}: {exc}"

    states = task["stages"]
    if all(s == "done" for s in states.values()):
        task["status"] = "done"
    elif states["report"] == "done":
        task["status"] = "partial"      # report exists but some stage failed
    else:
        task["status"] = "failed"


@app.post("/cases/{case_id}/process", status_code=202)
def process_case(case_id: str, background_tasks: BackgroundTasks):
    _case_or_404(case_id)
    ARTIFACTS[case_id] = {}             # start clean if processed again
    task_id = "t" + uuid.uuid4().hex[:8]
    TASKS[task_id] = {
        "task_id": task_id,
        "case_id": case_id,
        "status": "queued",
        "stages": {stage: "pending" for stage in STAGES},
        "errors": {},
    }
    background_tasks.add_task(run_pipeline, task_id)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="task not found")
    return TASKS[task_id]


@app.get("/cases/{case_id}/report")
def get_report(case_id: str):
    _case_or_404(case_id)
    report = ARTIFACTS[case_id].get("report")
    if report is None:
        raise HTTPException(status_code=409, detail="report not ready: call /process and wait for the task")
    return report


@app.get("/cases/{case_id}/artifacts/{stage}")
def get_artifact(case_id: str, stage: str):
    """Raw output of one stage (transcript, extraction, ...). The UI needs the
    transcript to show the evidence behind each report sentence."""
    _case_or_404(case_id)
    if stage not in STAGES:
        raise HTTPException(status_code=404, detail=f"unknown stage, use one of {STAGES}")
    data = ARTIFACTS[case_id].get(stage)
    if data is None:
        raise HTTPException(status_code=409, detail=f"{stage} output not available")
    return data
