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
