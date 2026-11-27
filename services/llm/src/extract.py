import json
import os
from typing import Optional

from mica_schemas import (
    Complaint, Extraction, ExtractionMeta, Finding, HistoryItem, Medication,
    Symptom, Transcript,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .grounding import verify

PROMPT_VERSION = "v1"
MAX_RETRIES = 2 


class LLMExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chief_complaint: Optional[Complaint] = None
    symptoms: list[Symptom] = Field(default_factory=list)
    denied_symptoms: list[Finding] = Field(default_factory=list)
    past_history: list[HistoryItem] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[HistoryItem] = Field(default_factory=list)
    family_history: list[HistoryItem] = Field(default_factory=list)
    social_history: list[HistoryItem] = Field(default_factory=list)


SYSTEM_PROMPT = """You extract structured medical information from a doctor-patient \
conversation in Egyptian Arabic (with English medical terms mixed in).
The text comes from speech recognition and may contain small transcription errors.

Return ONLY one JSON object that follows this JSON Schema:
{schema}

Rules:
1. Extract ONLY what was explicitly said. Never guess, never infer, never add.
2. Every item needs "evidence": a VERBATIM quote copied exactly from the segment text, \
and "segment_id": the number in brackets of the segment that contains it.
3. A symptom the patient denies ("مفيش سخونية") goes in "denied_symptoms", never in "symptoms".
4. A doctor's question is NOT evidence. Only record what the patient confirms or states \
(or what the doctor explicitly states as fact).
5. Use null for anything not stated (dose, severity, duration, onset...). Never invent a dose.
6. "name" is the standard English name ("chest pain"); "name_ar" is the Arabic as said.
7. Keep "duration" as said ("يومين"). If nothing is mentioned for a field, return [] (or null \
for chief_complaint).
8. Do not diagnose and do not suggest treatment.
9. "evidence" must be the SHORTEST verbatim span that supports that single item, \
not the whole sentence. Never include a negated or unrelated part in the evidence.
"""


def _seg(i, role, text):
    return {"id": i, "role": role, "text": text}


FEWSHOT = [
    ([_seg(0, "doctor", "إيه اللي تعبك؟"),
      _seg(1, "patient", "عندي صداع من تلات أيام."),
      _seg(2, "doctor", "بتاخد أدوية؟"),
      _seg(3, "patient", "باخد بنادول لما بحتاجه.")],
     {"chief_complaint": {"value": "صداع", "evidence": "عندي صداع من تلات أيام", "segment_id": 1},
      "symptoms": [{"name": "headache", "name_ar": "صداع", "present": True, "duration": "تلات أيام",
                    "evidence": "عندي صداع من تلات أيام", "segment_id": 1}],
      "medications": [{"name": "بنادول", "dose": None, "frequency": "لما بحتاجه",
                       "evidence": "باخد بنادول لما بحتاجه", "segment_id": 3}]}),
    ([_seg(0, "doctor", "عندك سخونية أو رعشة؟"),
      _seg(1, "patient", "لا مفيش سخونية بس عندي كحة من إمبارح.")],
     {"chief_complaint": {"value": "كحة", "evidence": "عندي كحة من إمبارح", "segment_id": 1},
      "symptoms": [{"name": "cough", "name_ar": "كحة", "present": True, "onset": "من إمبارح",
                    "evidence": "عندي كحة من إمبارح", "segment_id": 1}],
      "denied_symptoms": [{"name": "fever", "name_ar": "سخونية",
                           "evidence": "مفيش سخونية", "segment_id": 1}]}),
    ([_seg(0, "patient", "أنا عندي diabetes من سنتين وباخد metformin الصبح.")],
     {"past_history": [{"value": "diabetes", "duration": "سنتين",
                        "evidence": "عندي diabetes من سنتين", "segment_id": 0}],
      "medications": [{"name": "metformin", "dose": None, "frequency": "الصبح",
                       "evidence": "باخد metformin الصبح", "segment_id": 0}]}),
]


def format_transcript(segments) -> str:
    lines = []
    for s in segments:
        if isinstance(s, dict):
            lines.append(f"[{s['id']}] ({s['role']}): {s['text']}")
        else:
            lines.append(f"[{s.id}] ({s.role}): {s.text}")
    return "\n".join(lines)


def build_messages(tr: Transcript) -> list[dict]:
    schema = json.dumps(LLMExtraction.model_json_schema(), ensure_ascii=False)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT.replace("{schema}", schema)}]
    for segs, out in FEWSHOT:
        msgs.append({"role": "user", "content": format_transcript(segs)})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": format_transcript(tr.segments)})
    return msgs


def get_client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])


def call_llm(tr: Transcript, client=None, model: Optional[str] = None) -> LLMExtraction:
    client = client or get_client()
    model = model or os.environ["LLM_MODEL"]
    messages = build_messages(tr)
    last_err = None
    for _ in range(MAX_RETRIES + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            return LLMExtraction.model_validate_json(raw)
        except ValidationError as e:
            last_err = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "The JSON was invalid for the schema:\n"
                 f"{e}\nReturn the corrected JSON object only."},
            ]
    raise ValueError(f"LLM output failed validation after {MAX_RETRIES + 1} attempts: {last_err}")


def run_extraction(tr: Transcript, client=None, model: Optional[str] = None) -> Extraction:
    model = model or os.environ.get("LLM_MODEL")
    raw = call_llm(tr, client=client, model=model)
    ex = Extraction(
        case_id=tr.case_id,
        meta=ExtractionMeta(model=model, prompt_version=PROMPT_VERSION, temperature=0.0),
        **raw.model_dump(),
    )
    return verify(ex, tr)