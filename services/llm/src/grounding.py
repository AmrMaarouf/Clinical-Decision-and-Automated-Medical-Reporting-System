from mica_schemas import Extraction, Transcript, Dropped
from .normalize import ar_normalize
def quote_ok(quote:str, segment_text:str) -> bool:
    quote = ar_normalize(quote)
    segment_text = ar_normalize(segment_text)
    return quote!= "" and quote in segment_text
    


def verify(ex: Extraction, tr: Transcript) -> Extraction:
    segments = {segment.id : segment.text for segment in tr.segments}