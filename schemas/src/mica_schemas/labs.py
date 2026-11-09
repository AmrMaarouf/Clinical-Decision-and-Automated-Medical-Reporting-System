from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import MicaModel

Flag = Literal["low", "normal", "high", "unknown"]
RefSource = Literal["printed", "default", "none"]


class LabTest(MicaModel):
    name_raw: str                                   # exactly as read by OCR: "Hemoglobin"
    name_std: Optional[str] = None                  # standard name: "HGB"; None = not matched
    value: Optional[float] = None                   # None if it could not be read
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Flag = "unknown"
    ref_source: RefSource = "none"                  # printed on the report / default table / none
    ocr_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    needs_review: bool = False                      # low confidence or unclear row
    bbox: Optional[list[float]] = None              # [x1, y1, x2, y2] in the source image

    @model_validator(mode="after")
    def _check(self):
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        if self.ref_low is not None and self.ref_high is not None and self.ref_low > self.ref_high:
            raise ValueError("ref_low must be <= ref_high")
        has_ref = self.ref_low is not None or self.ref_high is not None
        if self.value is None or not has_ref:
            # no value or no range -> we do not guess a flag
            if self.flag != "unknown":
                raise ValueError("flag must be 'unknown' when value or reference range is missing")
            return self
        expected = "normal"
        if self.ref_low is not None and self.value < self.ref_low:
            expected = "low"
        elif self.ref_high is not None and self.value > self.ref_high:
            expected = "high"
        if self.flag != expected:
            raise ValueError(f"flag '{self.flag}' does not match value/range (expected '{expected}')")
        return self


class Labs(MicaModel):
    schema_version: str = "1.0"
    case_id: str
    source_file: str
    lab_name: Optional[str] = None
    report_date: Optional[str] = None
    tests: list[LabTest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)   # e.g. "row 7 unit missing"
