from src.grounding import quote_ok

def test_quote_ok():
    assert quote_ok("وجع في صدري", "عندي وجع في صدري من يومين.")
    assert not quote_ok("سخونية عالية", "عندي وجع في صدري من يومين.")