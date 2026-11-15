import re

def ar_normalize(text: str) -> str:
    #remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    #remove tatweel
    text = re.sub(r'\u0640', '', text)
    
    # convert Arabic letters to their standard forms
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text) 
    text = re.sub(r'\u0629', '\u0647', text)
    text = re.sub(r'\u0649', '\u064A', text) 
    text = re.sub(r'\u0671', '\u0627', text) 
    
    #remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    #remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    text = text.lower()
    return text