from mica_schemas import Extraction, Transcript, Dropped
from rapidfuzz import fuzz
from .normalize import ar_normalize

FUZZY_THRESHOLD = 90
MIN_FUZZY_LEN = 8
LIST_FIELDS = ("symptoms", "denied_symptoms", "past_history", "medications",
               "allergies", "family_history", "social_history")

def quote_ok(quote:str, segment_text:str) -> bool:
    q, s = ar_normalize(quote or ""), ar_normalize(segment_text or "")
    if not q or not s:
        return False
    if q in s:
        return True
    return len(q) >= MIN_FUZZY_LEN and fuzz.partial_ratio(q, s) >= FUZZY_THRESHOLD
    
def _reason(item, segments):
    if item.segment_id not in segments:
        return "segment_not_found"
    if not quote_ok(item.evidence, segments[item.segment_id]):
        return "evidence_not_in_segment"
    return None


def verify(ex: Extraction, tr: Transcript) -> Extraction:
    segments = {segment.id : segment.text for segment in tr.segments}
    dropped = list(ex.dropped)
    updates = {}

    if ex.chief_complaint is not None:
        r = _reason(ex.chief_complaint, segments)
        if r:
            dropped.append(Dropped(field="chief_complaint",
                                   quote=ex.chief_complaint.evidence, reason=r))
            updates["chief_complaint"] = None

    for name in LIST_FIELDS:
        kept = []
        for item in getattr(ex, name):
            r = _reason(item, segments)
            if r:
                dropped.append(Dropped(field=name, quote=item.evidence, reason=r))
            else:
                kept.append(item)
        updates[name] = kept
    not_mentioned = [n for n in ("chief_complaint", *LIST_FIELDS)
                     if not updates.get(n, getattr(ex, n))]
    updates["not_mentioned"] = not_mentioned
    updates["dropped"] = dropped
    return ex.model_copy(update=updates)

def hallucination_rate(before: Extraction, after: Extraction) -> float:
    total = sum(len(getattr(before, f)) for f in LIST_FIELDS)
    total += 1 if before.chief_complaint else 0
    new_drops = len(after.dropped) - len(before.dropped)
    return 100 * new_drops / total if total else 0.0

