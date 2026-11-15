from src.normalize import ar_normalize


def test_alef_forms():
    assert ar_normalize("إزاي") == ar_normalize("ازاي")


def test_punctuation_and_spaces():
    assert ar_normalize("عندي   وجع،  في صدري.") == "عندي وجع في صدري"


def test_punctuation_glued_words():
    assert ar_normalize("وجع،صدري") == "وجع صدري"


def test_english_lowercase():
    assert ar_normalize("Diabetes") == "diabetes"


def test_taa_marbuta():
    assert ar_normalize("الصحة") == ar_normalize("الصحه")