import pytest
from mica_schemas import Extraction, Transcript

from src.grounding import hallucination_rate, quote_ok, verify


# ------------------------------------------------------------ helpers
def make_transcript(texts):
    return Transcript.model_validate({
        "case_id": "t001",
        "segments": [
            {"id": i, "start": float(i), "end": float(i) + 0.5,
             "speaker": "SPEAKER_01", "role": "patient", "text": t}
            for i, t in enumerate(texts)
        ],
    })


TR = make_transcript([
    "أهلاً بحضرتك، إيه اللي تعبك؟",               # 0
    "عندي وجع في صدري من يومين.",                  # 1
    "عندك سخونية أو كحة؟",                         # 2
    "مفيش سخونية بس عندي شوية كحة.",               # 3
    "وبتشتكي من أمراض مزمنة؟",                     # 4
    "عندي ضغط عالي من سنتين.",                     # 5
    "وبتاخد أدوية؟",                               # 6
    "باخد كونكور كل يوم الصبح.",                   # 7
])


def make_extraction(**overrides):
    base = {
        "case_id": "t001",
        "chief_complaint": {"value": "وجع في الصدر",
                            "evidence": "عندي وجع في صدري من يومين", "segment_id": 1},
        "symptoms": [{"name": "cough", "name_ar": "كحة",
                      "evidence": "عندي شوية كحة", "segment_id": 3}],
        "denied_symptoms": [{"name": "fever", "name_ar": "سخونية",
                             "evidence": "مفيش سخونية", "segment_id": 3}],
        "past_history": [{"value": "hypertension", "duration": "سنتين",
                          "evidence": "عندي ضغط عالي من سنتين", "segment_id": 5}],
        "medications": [{"name": "كونكور", "frequency": "كل يوم الصبح",
                         "evidence": "باخد كونكور كل يوم الصبح", "segment_id": 7}],
    }
    base.update(overrides)
    return Extraction.model_validate(base)


# ------------------------------------------------------------ quote_ok
def test_exact_quote():
    assert quote_ok("وجع في صدري", "عندي وجع في صدري من يومين.")


def test_quote_not_in_segment():
    assert not quote_ok("سخونية عالية", "عندي وجع في صدري من يومين.")


def test_normalization_differences():
    # همزة + تشكيل + علامات ترقيم مختلفة
    assert quote_ok("إزاي حالك", "ازَّاي حالك؟")


def test_empty_or_none_quote():
    assert not quote_ok("", "عندي وجع")
    assert not quote_ok(None, "عندي وجع")
    assert not quote_ok("عندي وجع", "")


def test_fuzzy_whisper_typo_long_quote():
    # Whisper كتب 'ضعط' بدل 'ضغط'
    assert quote_ok("عندي ضغط عالي من سنتين", "عندي ضعط عالي من سنتين")


def test_fuzzy_not_applied_to_short_quote():
    # أقل من MIN_FUZZY_LEN: لازم يتطابق حرفيًا
    assert not quote_ok("ضغط", "ضعط")


def test_fuzzy_rejects_different_content():
    assert not quote_ok("باخد بروفين كل يوم", "باخد كونكور كل يوم الصبح")


# ------------------------------------------------------------ verify
def test_clean_extraction_keeps_everything():
    ex = make_extraction()
    out = verify(ex, TR)
    assert out.dropped == []
    assert out.chief_complaint is not None
    assert len(out.symptoms) == 1 and len(out.medications) == 1
    assert hallucination_rate(ex, out) == 0.0


def test_invented_symptom_is_dropped_and_logged():
    ex = make_extraction(symptoms=[
        {"name": "cough", "evidence": "عندي شوية كحة", "segment_id": 3},
        {"name": "fever", "evidence": "عندي سخونية عالية من إمبارح", "segment_id": 3},
    ])
    out = verify(ex, TR)
    assert [s.name for s in out.symptoms] == ["cough"]
    assert len(out.dropped) == 1
    d = out.dropped[0]
    assert d.field == "symptoms"
    assert d.reason == "evidence_not_in_segment"
    assert d.quote == "عندي سخونية عالية من إمبارح"


def test_unknown_segment_id_is_dropped():
    ex = make_extraction(medications=[
        {"name": "كونكور", "evidence": "باخد كونكور", "segment_id": 99},
    ])
    out = verify(ex, TR)
    assert out.medications == []
    assert out.dropped[0].reason == "segment_not_found"


def test_quote_in_wrong_segment_is_dropped():
    # الاقتباس موجود في الـ transcript بس مش في الـ segment المذكور
    ex = make_extraction(medications=[
        {"name": "كونكور", "evidence": "باخد كونكور كل يوم الصبح", "segment_id": 5},
    ])
    out = verify(ex, TR)
    assert out.medications == []
    assert out.dropped[0].reason == "evidence_not_in_segment"


def test_invalid_chief_complaint_is_removed():
    ex = make_extraction(chief_complaint={
        "value": "صداع", "evidence": "عندي صداع شديد", "segment_id": 1})
    out = verify(ex, TR)
    assert out.chief_complaint is None
    assert out.dropped[0].field == "chief_complaint"
    assert "chief_complaint" in out.not_mentioned


def test_not_mentioned_is_computed_by_code():
    out = verify(make_extraction(), TR)
    assert set(out.not_mentioned) == {"allergies", "family_history", "social_history"}


def test_field_emptied_by_verify_becomes_not_mentioned():
    ex = make_extraction(medications=[
        {"name": "x", "evidence": "باخد دوا مخترع", "segment_id": 7}])
    out = verify(ex, TR)
    assert "medications" in out.not_mentioned


def test_verify_does_not_mutate_input():
    ex = make_extraction(symptoms=[
        {"name": "fever", "evidence": "عندي سخونية عالية من إمبارح", "segment_id": 3}])
    verify(ex, TR)
    assert len(ex.symptoms) == 1 and ex.dropped == []


def test_existing_dropped_entries_are_preserved():
    ex = make_extraction(dropped=[{"field": "symptoms", "quote": "old", "reason": "old"}])
    out = verify(ex, TR)
    assert len(out.dropped) == 1 and out.dropped[0].quote == "old"


# ------------------------------------------------------------ hallucination_rate
def test_hallucination_rate_value():
    # 6 حقول (شكوى + عرضين + نفي + تاريخ + دوا)؛ واحد مخترع -> 1/6
    ex = make_extraction(symptoms=[
        {"name": "cough", "evidence": "عندي شوية كحة", "segment_id": 3},
        {"name": "fever", "evidence": "عندي سخونية عالية من إمبارح", "segment_id": 3},
    ])
    out = verify(ex, TR)
    assert hallucination_rate(ex, out) == pytest.approx(100 / 6)


def test_hallucination_rate_empty_extraction():
    ex = Extraction.model_validate({"case_id": "t001"})
    assert hallucination_rate(ex, verify(ex, TR)) == 0.0