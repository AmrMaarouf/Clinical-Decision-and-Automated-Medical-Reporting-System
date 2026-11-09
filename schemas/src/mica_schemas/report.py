from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import MicaModel

NOT_MENTIONED_AR = "غير مذكور في الجلسة"

Status = Literal["draft", "reviewed", "signed"]
GeneratedBy = Literal["code", "llm"]


class Section(MicaModel):
    text: str
    # where the content came from, e.g. "seg:1", "extraction.chief_complaint", "labs", "xray"
    sources: list[str] = Field(default_factory=list)
    generated_by: GeneratedBy = "code"
    requires_review: bool = False

    @model_validator(mode="after")
    def _no_claim_without_source(self):
        if self.text.strip() != NOT_MENTIONED_AR and not self.sources:
            raise ValueError("a section with content must list its sources")
        if self.text.strip() == NOT_MENTIONED_AR and self.sources:
            raise ValueError("a 'not mentioned' section must have no sources")
        return self


class Sections(MicaModel):
    """The 7 fixed sections of the Arabic report."""

    chief_complaint: Section
    hpi: Section                                    # history of present illness
    past_history: Section                           # history, medications, allergies
    examination: Section
    investigations: Section                         # labs + x-ray
    assessment: Section
    plan: Section

    @model_validator(mode="after")
    def _assessment_needs_review(self):
        if not self.assessment.requires_review:
            raise ValueError("assessment must have requires_review=True")
        return self


class UnsupportedClaim(MicaModel):
    section: str
    text: str
    reason: str


class ReportMeta(MicaModel):
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class Report(MicaModel):
    schema_version: str = "1.0"
    case_id: str
    language: str = "ar"
    status: Status = "draft"
    sections: Sections
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    meta: Optional[ReportMeta] = None
