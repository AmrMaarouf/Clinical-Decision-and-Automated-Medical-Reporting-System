"""Fake pipeline stages.

Each stage returns the matching fixture (validated against its schema) instead
of running a real model. When a teammate's real service is ready, the real call
goes where `NotImplementedError` is raised below, and nothing else changes.
"""
import json
import os
import time
from pathlib import Path

from mica_schemas import Extraction, Labs, Report, Transcript, Xray

# Execution order. A stage only runs after the ones it depends on (see main.py).
STAGES = ["asr", "extract", "labs", "xray", "report"]

# stage -> (fixture file, schema model, env flag that turns fake mode on/off)
STAGE_CONFIG = {
    "asr": ("transcript.json", Transcript, "ASR_FAKE"),
    "extract": ("extraction.json", Extraction, "LLM_FAKE"),
    "labs": ("labs.json", Labs, "LABS_FAKE"),
    "xray": ("xray.json", Xray, "XRAY_FAKE"),
    "report": ("report.json", Report, "LLM_FAKE"),
}


def _fixtures_dir() -> Path:
    """FIXTURES_DIR if set, else look for fixtures/v1 in the parents of this file
    (works both inside Docker: /app/fixtures, and locally: <repo>/fixtures)."""
    env = os.getenv("FIXTURES_DIR")
    if env:
        return Path(env)
    for base in Path(__file__).resolve().parents:
        candidate = base / "fixtures" / "v1"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("fixtures/v1 not found; set FIXTURES_DIR")


def is_fake(stage: str) -> bool:
    flag = STAGE_CONFIG[stage][2]
    return os.getenv(flag, "1") == "1"


def run_stage(stage: str, case_id: str) -> dict:
    """Run one stage and return its JSON output (a plain dict)."""
    if stage not in STAGE_CONFIG:
        raise ValueError(f"unknown stage: {stage}")
    filename, model, flag = STAGE_CONFIG[stage]

    if not is_fake(stage):
        # TODO: call the real service here (httpx -> http://asr:8000/run, ...)
        raise NotImplementedError(f"{stage}: real service not connected yet (set {flag}=1)")

    # pretend the model takes time, so polling /tasks/{id} shows progress
    time.sleep(float(os.getenv("FAKE_DELAY_S", "1")))

    data = json.loads((_fixtures_dir() / filename).read_text(encoding="utf-8"))
    data["case_id"] = case_id
    # validating here means a broken fixture or schema fails loudly, like a real service would
    return model.model_validate(data).model_dump(mode="json")
