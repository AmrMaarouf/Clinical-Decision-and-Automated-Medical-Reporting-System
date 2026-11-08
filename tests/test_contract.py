"""Contract tests: every fixture must validate against its schema, and the
extraction fixture must be grounded in the transcript fixture."""
import json
from pathlib import Path

import pytest
from mica_schemas import Extraction, Transcript

FX = Path(__file__).resolve().parents[1] / "fixtures" / "v1"


def load(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def test_transcript_fixture_is_valid():
    Transcript.model_validate(load("transcript.json"))


def test_extraction_fixture_is_valid():
    Extraction.model_validate(load("extraction.json"))


def test_extraction_evidence_is_in_transcript():
    t = Transcript.model_validate(load("transcript.json"))
    e = Extraction.model_validate(load("extraction.json"))
    text = {s.id: s.text for s in t.segments}
    items = []
    if e.chief_complaint:
        items.append(e.chief_complaint)
    for group in (e.symptoms, e.denied_symptoms, e.past_history,
                  e.medications, e.allergies, e.family_history, e.social_history):
        items.extend(group)
    assert items
    for it in items:
        assert it.segment_id in text, f"unknown segment {it.segment_id}"
        assert it.evidence in text[it.segment_id], f"evidence not in segment: {it.evidence!r}"


def test_unknown_field_is_rejected():
    bad = load("transcript.json")
    bad["segments"][0]["speeker"] = "typo"
    with pytest.raises(Exception):
        Transcript.model_validate(bad)


def test_bad_role_is_rejected():
    bad = load("transcript.json")
    bad["segments"][0]["role"] = "nurse"
    with pytest.raises(Exception):
        Transcript.model_validate(bad)


def test_evidence_cannot_be_empty():
    bad = load("extraction.json")
    bad["chief_complaint"]["evidence"] = ""
    with pytest.raises(Exception):
        Extraction.model_validate(bad)
