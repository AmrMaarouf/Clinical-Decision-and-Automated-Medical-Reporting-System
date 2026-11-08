#!/usr/bin/env bash
# MICA: writes the first schemas (transcript + extraction), fixtures, and contract tests.
# Run from the ROOT of the repo:   bash add_schemas.sh
# It OVERWRITES these files (they are new).
set -euo pipefail
w() { mkdir -p "$(dirname "$1")"; cat > "$1"; echo "wrote: $1"; }

w schemas/src/mica_schemas/_base.py <<'MICA_EOF'
from pydantic import BaseModel, ConfigDict


class MicaModel(BaseModel):
    """Base for every MICA schema. Unknown fields are rejected on purpose:
    a typo or an undeclared field fails at the service boundary."""

    model_config = ConfigDict(extra="forbid")
MICA_EOF

w schemas/src/mica_schemas/transcript.py <<'MICA_EOF'
from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import MicaModel

Role = Literal["doctor", "patient", "unknown"]


class Segment(MicaModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)          # seconds
    end: float = Field(ge=0)            # seconds
    speaker: str                        # raw diarization label, e.g. "SPEAKER_00"
    role: Role = "unknown"
    text: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.end < self.start:
            raise ValueError("segment end must be >= start")
        return self


class Transcript(MicaModel):
    schema_version: str = "1.0"
    case_id: str
    language: str = "ar-EG"
    duration_s: Optional[float] = Field(default=None, ge=0)
    segments: list[Segment]
    role_method: Optional[str] = None   # e.g. "heuristic+llm"
MICA_EOF

w schemas/src/mica_schemas/extraction.py <<'MICA_EOF'
from typing import Optional

from pydantic import Field

from ._base import MicaModel


class Grounded(MicaModel):
    """Anything extracted from the conversation must point back to its source."""

    evidence: str = Field(min_length=1)   # verbatim quote from the transcript
    segment_id: int = Field(ge=0)         # which segment the quote is in


class Complaint(Grounded):
    value: str


class Finding(Grounded):
    name: str                             # standard name, English: "fever"
    name_ar: Optional[str] = None         # as said, Arabic: "سخونية"


class Symptom(Finding):
    present: bool = True
    duration: Optional[str] = None        # as said: "يومين"
    severity: Optional[str] = None
    onset: Optional[str] = None


class HistoryItem(Grounded):
    value: str
    duration: Optional[str] = None


class Medication(Grounded):
    name: str
    dose: Optional[str] = None            # null unless explicitly said
    frequency: Optional[str] = None


class Dropped(MicaModel):
    """A field removed by the grounding check (used to measure hallucination)."""

    field: str
    quote: Optional[str] = None
    reason: str


class ExtractionMeta(MicaModel):
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    temperature: Optional[float] = None


class Extraction(MicaModel):
    schema_version: str = "1.0"
    case_id: str
    chief_complaint: Optional[Complaint] = None
    symptoms: list[Symptom] = Field(default_factory=list)          # present ones
    denied_symptoms: list[Finding] = Field(default_factory=list)   # "مفيش سخونية"
    past_history: list[HistoryItem] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[HistoryItem] = Field(default_factory=list)
    family_history: list[HistoryItem] = Field(default_factory=list)
    social_history: list[HistoryItem] = Field(default_factory=list)
    not_mentioned: list[str] = Field(default_factory=list)         # computed by code
    dropped: list[Dropped] = Field(default_factory=list)           # grounding log
    meta: Optional[ExtractionMeta] = None
MICA_EOF

w schemas/src/mica_schemas/__init__.py <<'MICA_EOF'
from .extraction import (
    Complaint,
    Dropped,
    Extraction,
    ExtractionMeta,
    Finding,
    Grounded,
    HistoryItem,
    Medication,
    Symptom,
)
from .transcript import Segment, Transcript

__all__ = [
    "Complaint", "Dropped", "Extraction", "ExtractionMeta", "Finding", "Grounded",
    "HistoryItem", "Medication", "Symptom", "Segment", "Transcript",
]
MICA_EOF

w fixtures/v1/transcript.json <<'MICA_EOF'
{
  "schema_version": "1.0",
  "case_id": "demo001",
  "language": "ar-EG",
  "duration_s": 42.0,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.0,
      "speaker": "SPEAKER_00",
      "role": "doctor",
      "text": "أهلاً بحضرتك، إيه اللي تعبك؟",
      "confidence": 0.95
    },
    {
      "id": 1,
      "start": 4.5,
      "end": 9.0,
      "speaker": "SPEAKER_01",
      "role": "patient",
      "text": "عندي وجع في صدري من يومين.",
      "confidence": 0.93
    },
    {
      "id": 2,
      "start": 9.5,
      "end": 12.0,
      "speaker": "SPEAKER_00",
      "role": "doctor",
      "text": "عندك سخونية أو كحة؟",
      "confidence": 0.95
    },
    {
      "id": 3,
      "start": 12.5,
      "end": 17.0,
      "speaker": "SPEAKER_01",
      "role": "patient",
      "text": "مفيش سخونية بس عندي شوية كحة.",
      "confidence": 0.92
    },
    {
      "id": 4,
      "start": 17.5,
      "end": 21.0,
      "speaker": "SPEAKER_00",
      "role": "doctor",
      "text": "وبتشتكي من أمراض مزمنة؟",
      "confidence": 0.94
    },
    {
      "id": 5,
      "start": 21.5,
      "end": 25.0,
      "speaker": "SPEAKER_01",
      "role": "patient",
      "text": "عندي ضغط عالي من سنتين.",
      "confidence": 0.94
    },
    {
      "id": 6,
      "start": 25.5,
      "end": 28.0,
      "speaker": "SPEAKER_00",
      "role": "doctor",
      "text": "وبتاخد أدوية؟",
      "confidence": 0.96
    },
    {
      "id": 7,
      "start": 28.5,
      "end": 33.0,
      "speaker": "SPEAKER_01",
      "role": "patient",
      "text": "باخد كونكور كل يوم الصبح.",
      "confidence": 0.9
    }
  ],
  "role_method": "synthetic-fixture"
}
MICA_EOF

w fixtures/v1/extraction.json <<'MICA_EOF'
{
  "schema_version": "1.0",
  "case_id": "demo001",
  "chief_complaint": {
    "value": "وجع في الصدر",
    "evidence": "عندي وجع في صدري من يومين",
    "segment_id": 1
  },
  "symptoms": [
    {
      "name": "chest pain",
      "name_ar": "وجع في الصدر",
      "present": true,
      "duration": "يومين",
      "severity": null,
      "onset": null,
      "evidence": "عندي وجع في صدري من يومين",
      "segment_id": 1
    },
    {
      "name": "cough",
      "name_ar": "كحة",
      "present": true,
      "duration": null,
      "severity": null,
      "onset": null,
      "evidence": "عندي شوية كحة",
      "segment_id": 3
    }
  ],
  "denied_symptoms": [
    {
      "name": "fever",
      "name_ar": "سخونية",
      "evidence": "مفيش سخونية",
      "segment_id": 3
    }
  ],
  "past_history": [
    {
      "value": "hypertension",
      "duration": "سنتين",
      "evidence": "عندي ضغط عالي من سنتين",
      "segment_id": 5
    }
  ],
  "medications": [
    {
      "name": "كونكور",
      "dose": null,
      "frequency": "كل يوم الصبح",
      "evidence": "باخد كونكور كل يوم الصبح",
      "segment_id": 7
    }
  ],
  "allergies": [],
  "family_history": [],
  "social_history": [],
  "not_mentioned": [
    "severity",
    "allergies",
    "family_history",
    "social_history"
  ],
  "dropped": [],
  "meta": null
}
MICA_EOF

w fixtures/README.md <<'MICA_EOF'
# fixtures

Fixed example files that every service develops and tests against.

| file | what it is |
|---|---|
| `v1/transcript.json` | **Synthetic** short doctor/patient dialogue written by hand. It is NOT a real recording and NOT real patient data. |
| `v1/extraction.json` | The expected extraction for that dialogue (hand-written gold). |

Rules: never commit real patient data. Anything derived from a real source must be
de-identified first, and its source must be written in this file.
MICA_EOF

w tests/test_contract.py <<'MICA_EOF'
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
MICA_EOF

echo
echo "Next:"
echo "  docker compose build llm"
echo "  docker compose run --rm -v "\$PWD/fixtures:/app/fixtures:ro" -v "\$PWD/tests:/app/contract_tests:ro" llm python -m pytest /app/contract_tests -q -p no:cacheprovider"
