"""Contract tests: every fixture must validate against its schema, and the
fixtures must be consistent with each other."""
import copy
import json
from pathlib import Path

import pytest
from mica_schemas import (
    NOT_MENTIONED_AR, Extraction, Labs, Report, Transcript, Xray,
)

FX = Path(__file__).resolve().parents[1] / "fixtures" / "v1"


def load(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- valid fixtures
@pytest.mark.parametrize("name,model", [
    ("transcript.json", Transcript),
    ("extraction.json", Extraction),
    ("labs.json", Labs),
    ("xray.json", Xray),
    ("report.json", Report),
])
def test_fixture_is_valid(name, model):
    model.model_validate(load(name))


def test_all_fixtures_share_case_id():
    ids = {load(n)["case_id"] for n in
           ("transcript.json", "extraction.json", "labs.json", "xray.json", "report.json")}
    assert len(ids) == 1


# ------------------------------------------------------------ cross-file checks
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


def test_report_segment_sources_exist_in_transcript():
    t = Transcript.model_validate(load("transcript.json"))
    r = Report.model_validate(load("report.json"))
    ids = {s.id for s in t.segments}
    for name, sec in r.sections.model_dump().items():
        for src in sec["sources"]:
            if src.startswith("seg:"):
                assert int(src.split(":")[1]) in ids, f"{name}: unknown {src}"


def test_report_has_exactly_seven_sections():
    assert len(Report.model_validate(load("report.json")).sections.model_dump()) == 7


# ----------------------------------------------------- transcript / extraction
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


# ------------------------------------------------------------------------ labs
def test_labs_wrong_flag_is_rejected():
    bad = load("labs.json")
    bad["tests"][0]["flag"] = "normal"          # 10.2 with range 12-16 is low
    with pytest.raises(Exception):
        Labs.model_validate(bad)


def test_labs_flag_must_be_unknown_without_range():
    bad = load("labs.json")
    bad["tests"][0].update(ref_low=None, ref_high=None, ref_source="none")  # flag still "low"
    with pytest.raises(Exception):
        Labs.model_validate(bad)


def test_labs_unmatched_name_is_allowed():
    ok = load("labs.json")
    ok["tests"][0].update(name_std=None, needs_review=True)
    Labs.model_validate(ok)


def test_labs_bad_flag_value_is_rejected():
    bad = load("labs.json")
    bad["tests"][0]["flag"] = "very high"
    with pytest.raises(Exception):
        Labs.model_validate(bad)


# ------------------------------------------------------------------------ xray
def test_xray_probabilities_must_sum_to_one():
    bad = load("xray.json")
    bad["probabilities"] = {"normal": 0.5, "pneumonia": 0.9}
    with pytest.raises(Exception):
        Xray.model_validate(bad)


def test_xray_bad_label_is_rejected():
    bad = load("xray.json")
    bad["prediction"]["label"] = "cancer"
    with pytest.raises(Exception):
        Xray.model_validate(bad)


def test_xray_disclaimer_cannot_be_empty():
    bad = load("xray.json")
    bad["disclaimer"] = "  "
    with pytest.raises(Exception):
        Xray.model_validate(bad)


# ---------------------------------------------------------------------- report
def test_report_section_with_content_needs_sources():
    bad = copy.deepcopy(load("report.json"))
    bad["sections"]["hpi"]["sources"] = []
    with pytest.raises(Exception):
        Report.model_validate(bad)


def test_report_not_mentioned_section_must_have_no_sources():
    bad = copy.deepcopy(load("report.json"))
    bad["sections"]["plan"]["sources"] = ["seg:1"]
    assert bad["sections"]["plan"]["text"] == NOT_MENTIONED_AR
    with pytest.raises(Exception):
        Report.model_validate(bad)


def test_report_assessment_must_require_review():
    bad = copy.deepcopy(load("report.json"))
    bad["sections"]["assessment"]["requires_review"] = False
    with pytest.raises(Exception):
        Report.model_validate(bad)


def test_report_bad_status_is_rejected():
    bad = copy.deepcopy(load("report.json"))
    bad["status"] = "approved"
    with pytest.raises(Exception):
        Report.model_validate(bad)


def test_report_missing_section_is_rejected():
    bad = copy.deepcopy(load("report.json"))
    del bad["sections"]["plan"]
    with pytest.raises(Exception):
        Report.model_validate(bad)
