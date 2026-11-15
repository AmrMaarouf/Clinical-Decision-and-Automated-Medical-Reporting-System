def quote_ok(quote:str, segment_text:str) -> bool:
    if quote in segment_text:
        return True
    else:
        return False
    

quote_ok("وجع في صدري", "عندي وجع في صدري من يومين.")
quote_ok("سخونية عالية", "عندي وجع في صدري من يومين.")