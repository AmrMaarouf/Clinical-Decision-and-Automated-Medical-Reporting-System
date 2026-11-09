from typing import Literal, Optional

from pydantic import Field, model_validator

from ._base import MicaModel

DISCLAIMER_AR = "ملاحظة مساعدة آلية، مش تشخيص، ومحتاجة مراجعة أخصائي أشعة."

View = Literal["AP", "PA", "unknown"]
XrayLabel = Literal["normal", "pneumonia", "uncertain"]


class ModelInfo(MicaModel):
    name: str                                       # e.g. "densenet121-finetuned"
    version: str


class Prediction(MicaModel):
    label: XrayLabel                                # "uncertain" when probability is near 0.5
    probability: float = Field(ge=0, le=1)          # probability of the predicted class


class Xray(MicaModel):
    schema_version: str = "1.0"
    case_id: str
    image_file: str
    view: View = "unknown"
    model: ModelInfo
    prediction: Prediction
    probabilities: dict[str, float]                 # {"normal": 0.19, "pneumonia": 0.81}
    gradcam_file: Optional[str] = None
    disclaimer: str = DISCLAIMER_AR
    quality_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        for k, v in self.probabilities.items():
            if not 0 <= v <= 1:
                raise ValueError(f"probability for '{k}' must be in [0, 1]")
        if self.probabilities and abs(sum(self.probabilities.values()) - 1.0) > 0.02:
            raise ValueError("probabilities must sum to 1 (+/- 0.02)")
        if not self.disclaimer.strip():
            raise ValueError("disclaimer cannot be empty")
        return self
