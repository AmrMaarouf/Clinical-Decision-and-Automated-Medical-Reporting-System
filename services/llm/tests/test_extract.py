import json
from types import SimpleNamespace

import pytest
from mica_schemas import Transcript

from src.extract import MAX_RETRIES, build_messages, run_extraction

TR = Transcript.model_validate({
    "case_id": "t001",
    "segments": [
        {"id": 0, "start": 0, "end": 1, "speaker": "S0", "role": "doctor",
         "text": "عندك سخونية؟"},
        {"id": 1, "start": 1, "end": 2, "speaker": "S1", "role": "patient",
         "text": "مفيش سخونية بس عندي كحة من إمبارح."},
    ],
})

GOOD = {
    "chief_complaint": {"value": "كحة", "evidence": "عندي كحة من إمبارح", "segment_id": 1},
    "symptoms": [{"name": "cough", "evidence": "عندي كحة من إمبارح", "segment_id": 1}],
    "denied_symptoms": [{"name": "fever", "evidence": "مفيش سخونية", "segment_id": 1}],
}


class FakeClient:
    """Returns the given raw strings one by one, like chat.completions.create."""

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


def test_happy_path_sets_meta_and_keeps_grounded_fields():
    c = FakeClient(json.dumps(GOOD, ensure_ascii=False))
    ex = run_extraction(TR, client=c, model="my-deployment")
    assert ex.chief_complaint.value == "كحة"
    assert ex.dropped == []
    assert ex.meta.model == "my-deployment" and ex.meta.temperature == 0.0
    assert c.calls[0]["temperature"] == 0 and c.calls[0]["model"] == "my-deployment"
    assert set(ex.not_mentioned) == {"past_history", "medications", "allergies",
                                     "family_history", "social_history"}


def test_hallucinated_field_is_dropped_by_grounding():
    bad = dict(GOOD, medications=[
        {"name": "x", "evidence": "باخد دوا مخترع", "segment_id": 1}])
    c = FakeClient(json.dumps(bad, ensure_ascii=False))
    ex = run_extraction(TR, client=c, model="m")
    assert ex.medications == []
    assert ex.dropped[0].field == "medications"


def test_retry_after_invalid_json():
    c = FakeClient("not json at all", json.dumps(GOOD, ensure_ascii=False))
    ex = run_extraction(TR, client=c, model="m")
    assert ex.chief_complaint is not None and len(c.calls) == 2
    assert "invalid" in c.calls[1]["messages"][-1]["content"]


def test_gives_up_after_max_retries():
    c = FakeClient(*["{}garbage"] * (MAX_RETRIES + 1))
    with pytest.raises(ValueError):
        run_extraction(TR, client=c, model="m")
    assert len(c.calls) == MAX_RETRIES + 1


def test_llm_cannot_set_code_computed_fields():
    bad = dict(GOOD, not_mentioned=["x"])
    c = FakeClient(json.dumps(bad), json.dumps(GOOD, ensure_ascii=False))
    run_extraction(TR, client=c, model="m")
    assert len(c.calls) == 2     # extra field rejected -> retry


def test_messages_contain_schema_fewshot_and_transcript():
    msgs = build_messages(TR)
    assert msgs[0]["role"] == "system" and "JSON" in msgs[0]["content"]
    assert len(msgs) == 1 + 3 * 2 + 1    # system + 3 few-shot pairs + real input
    assert msgs[-1]["content"].startswith("[0] (doctor)")
