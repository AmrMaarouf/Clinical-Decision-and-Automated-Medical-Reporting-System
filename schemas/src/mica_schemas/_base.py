from pydantic import BaseModel, ConfigDict


class MicaModel(BaseModel):
    """Base for every MICA schema. Unknown fields are rejected on purpose:
    a typo or an undeclared field fails at the service boundary."""

    model_config = ConfigDict(extra="forbid")
