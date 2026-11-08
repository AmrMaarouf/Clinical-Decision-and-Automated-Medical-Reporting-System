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
