#!/usr/bin/env python
# coding: utf-8

# # 🩺 Doctor–Patient Egyptian Arabic Transcription (Kaggle)
# 
# **Pipeline:** تنظيف صوت خفيف ← faster-whisper ← pyannote (يحدد عدد المتكلمين تلقائياً) ← قاموس مصطلحات إنجليزي ← تصحيح LLM اختياري (Claude أو Qwen) ← تحديد مين الدكتور ومين المريض ← JSON
# 
# ## قبل ما تبدأي (مرة واحدة)
# 1. **Settings (يمين):** Accelerator = **GPU T4 x2** (أو P100) ، و **Internet = On**.
# 2. **Add-ons ← Secrets:** ضيفي `HF_TOKEN` (من Hugging Face). ولو هتستخدمي Claude ضيفي كمان `ANTHROPIC_API_KEY`.
# 3. وافقي على شروط الموديلين دول على Hugging Face (Agree and access):
#    - https://huggingface.co/pyannote/speaker-diarization-3.1
#    - https://huggingface.co/pyannote/segmentation-3.0
# 
# بعد كده شغّلي الخلايا بالترتيب. **الريكوردات بترفعيها من خلية الرفع، مش محتاجة Dataset.**

# ## 1) تثبيت المكتبات

# In[ ]:


get_ipython().system('nvidia-smi -L')
# torch/torchaudio لازم 2.8.0 بالظبط: torchaudio 2.9+ شالت AudioMetaData وبتكسر pyannote
get_ipython().system('pip install -q "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0"')
get_ipython().system('pip install -q -U faster-whisper "pyannote.audio==3.4.0" anthropic ipywidgets')
get_ipython().system('pip install -q -U bitsandbytes accelerate "transformers>=4.45,<5" "huggingface_hub<1.0" "numpy<2.3" "pytorch-metric-learning<3"')
get_ipython().system('ffmpeg -version | head -1')


# ### ⚠️ مهم: بعد ما الخلية اللي فوق تخلص، اعملي **Restart Session** (من قايمة Run أو Kernel) وبعدها كمّلي من الخلية اللي بعدها. (من غير الـ Restart الإصدارات القديمة هتفضل شغالة.)

# In[1]:


import torch, torchaudio
print("torch:", torch.__version__, "| torchaudio:", torchaudio.__version__)
import huggingface_hub
print("huggingface_hub:", huggingface_hub.__version__)
assert int(huggingface_hub.__version__.split(".")[0]) < 1, "huggingface_hub لازم يكون أقل من 1.0: شغّلي خلية التثبيت تاني وبعدها Restart"
assert hasattr(torchaudio, "AudioMetaData"), "لسه torchaudio الجديد شغال: اعملي Restart Session وشغّلي الخلية دي تاني"
from pyannote.audio import Pipeline
print("pyannote import OK")


# ## 2) الإعدادات (دي الخلية الوحيدة اللي ممكن تعدّلي فيها)

# In[2]:


import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"   # توافق pyannote مع نسخ torch الجديدة

# ---------- المسارات ----------
RECORDS_DIR = "/kaggle/working/records"     # فيه الريكوردات (خلية الرفع بتحطها هنا)
OUT_DIR     = "/kaggle/working/results"
TMP_DIR     = "/kaggle/working/tmp"

# ---------- Whisper ----------
WHISPER_MODEL = "large-v3"      # الأدق. بدائل: "large-v3-turbo" (أسرع) ، "medium" (أخف)
LANGUAGE      = "ar"
BEAM_SIZE     = 5

# ---------- إزالة النويز ----------
NOISE = "light"                 # "off" | "light" | "medium"   (light = بسيط جداً زي ما طلبتي)

# ---------- عدد المتكلمين ----------
MIN_SPEAKERS = 2                # الموديل بيحدد العدد لوحده بين الحدّين دول
MAX_SPEAKERS = 4                # لو متأكدة إنهم اتنين بس: خليهم الاتنين 2
MIN_SPEAKER_SECONDS = 2.0       # أي متكلم كلامه أقل من كده بيتعامل كغلطة وبيتدمج
SPEAKER_MODE = "hybrid"          # "audio" = بالصوت بس | "text" = بالمعنى بس | "hybrid" = الصوت كتلميح + المعنى بيحسم (الأفضل، محتاج LLM). لو LLM_BACKEND = "none" استخدمي "audio"

# ---------- الـ LLM (تصحيح النص + تحديد الدكتور/المريض) ----------
LLM_BACKEND  = "qwen"         # "none" | "claude" | "qwen"
CLAUDE_MODEL = "claude-sonnet-5-5"        # بيتبعت النص لسيرفر خارجي
QWEN_MODEL   = "Qwen/Qwen3-14B"   # مجاني: بيشتغل جوه Kaggle (4-bit). بدائل: "Qwen/Qwen3-8B" (أخف وأسرع) | "Qwen/Qwen2.5-14B-Instruct"
LLM_BATCH    = 8                # عدد الجمل في كل طلب للـ LLM

# ---------- قاموسك الخاص (يضاف على القاموس الجاهز) ----------
CUSTOM_TERMS    = {    

    "كونكور": "Concor", "بانادول": "Panadol", "بروفين": "Brufen", "بروفينه": "Brufen",
    "كتافلام": "Cataflam", "فولتارين": "Voltaren", "ديكلوفيناك": "diclofenac",
    "اوجمنتين": "Augmentin", "اوجمانتين": "Augmentin", "اموكسيل": "Amoxil", "زيثروماكس": "Zithromax",
    "فلاجيل": "Flagyl", "انتينال": "Antinal", "بوسكوبان": "Buscopan",
    "جلوكوفاج": "Glucophage", "ميتفورمين": "metformin",
    "نيكسيوم": "Nexium", "كونترولوك": "Controloc", "فنتولين": "Ventolin",
    "ليبيتور": "Lipitor", "اتورفاستاتين": "atorvastatin", "كلكسان": "Clexane", "بلافيكس": "Plavix",
    "كوزار": "Cozaar", "نورفاسك": "Norvasc", "ديجوكسين": "digoxin", "بريدنيزولون": "prednisolone",
    "كلاريتين": "Claritin", "زيرتك": "Zyrtec", "بيتادين": "Betadine",
    "الكلمة كما بتتكتب بالعربي": "English"
    # "كونكور": "Concor",
}
CUSTOM_HOTWORDS = []           # كلمات إنجليزي مهمة تساعد Whisper يسمعها صح
print("config OK")


# ## 3) الـ Tokens

# In[3]:


import os, getpass
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()

def get_secret(name, required=True):
    try:
        return sec.get_secret(name)
    except Exception:
        if required:
            return getpass.getpass(f"{name} not found in Secrets. Paste it here: ")
        return None

HF_TOKEN = get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
if LLM_BACKEND == "claude":
    os.environ["ANTHROPIC_API_KEY"] = get_secret("ANTHROPIC_API_KEY")
print("tokens OK")


# ## 4) ارفعي الريكوردات من جهازك
# دوسي الزرار واختاري ملف أو أكتر، وبعدها **شغّلي الخلية اللي بعدها** عشان تتحفظ.

# In[30]:


import os
import ipywidgets as w
from IPython.display import display
os.makedirs(RECORDS_DIR, exist_ok=True)
uploader = w.FileUpload(accept="audio/*,video/mp4,.mp3,.wav,.m4a,.ogg,.flac,.aac,.opus", multiple=True)
display(uploader)


# In[31]:


# شغّليها بعد ما تختاري الملفات
v = uploader.value
items = list(v.values()) if isinstance(v, dict) else list(v)

if not items:
    raise ValueError("Please upload at least one recording.")

# Clear previous recordings
import shutil

shutil.rmtree(RECORDS_DIR, ignore_errors=True)
os.makedirs(RECORDS_DIR, exist_ok=True)

# Clear previous JSON results
shutil.rmtree(OUT_DIR, ignore_errors=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Save new recordings only
for it in items:
    name = it["name"] if "name" in it else it["metadata"]["name"]
    data = bytes(it["content"])

    with open(os.path.join(RECORDS_DIR, name), "wb") as f:
        f.write(data)

print("New files in", RECORDS_DIR, ":", os.listdir(RECORDS_DIR))


# ## 5) القاموس والـ Prompt

# In[32]:


import re, json, difflib, gc, subprocess, shutil
from pathlib import Path

# كلمة بالعربي (زي ما Whisper ممكن يكتبها) -> الإنجليزي الصح
MEDICAL_DICT = {
    # ---- (من عندك) تحاليل وأشعة ----
    "سي بي سي": "CBC", "سى بى سى": "CBC", "اكس راي": "X-ray", "إكس راي": "X-ray", "اكسراي": "X-ray",
    "ام ار اي": "MRI", "ام آر آي": "MRI", "إم آر آي": "MRI", "سي تي": "CT", "اي سي جي": "ECG",
    "إي سي جي": "ECG", "ايكو": "Echo", "إيكو": "Echo", "ايه وان سي": "HbA1c",
    "اتش بي ايه وان سي": "HbA1c", "كوليسترول": "cholesterol", "كولسترول": "cholesterol",
    "تراي جلسرايد": "triglycerides", "تراى جليسرايد": "triglycerides", "فيتامين دي": "Vitamin D",
    "فيتامين بي 12": "Vitamin B12", "فيتامين بي١٢": "Vitamin B12", "كرياتينين": "creatinine",
    "يوريا": "urea", "ال دي ال": "LDL", "اتش دي ال": "HDL", "تي اس اتش": "TSH", "سي ار بي": "CRP",
    "اي سي يو": "ICU",
    # ---- (من عندك) أمراض وحالات ----
    "هايبرتنشن": "hypertension", "هايبرتنشان": "hypertension", "دايابيتس": "diabetes",
    "دياباتس": "diabetes", "ديابيتس": "diabetes", "انيميا": "anemia", "أنيميا": "anemia",
    "اسثما": "asthma", "ازما": "asthma", "ميجرين": "migraine", "ميغرين": "migraine",
    "اليرجي": "allergy", "أليرجي": "allergy", "انفلونزا": "flu", "اي بي اس": "IBS", "جي ار دي": "GERD",
    "ثايرويد": "thyroid", "سيروم": "serum",
    # ---- (من عندك) مكملات ----
    "فيتامين": "vitamin", "بروبيوتيك": "probiotic", "اوميجا ثري": "Omega-3", "أوميجا 3": "Omega-3",
    # ---- (من عندك) إضافات من التجارب ----
    "بليفايك": "pelvic", "ألتراسوند": "ultrasound", "التراسوند": "ultrasound",
    "اندومتريس": "endometriosis", "اندوميتريوسيس": "endometriosis", "MII": "MRI",
    # ---- أشعة وفحوصات ----
    "سونار": "ultrasound", "سونار دوبلر": "Doppler ultrasound", "دوبلر": "Doppler",
    "سي تي سكان": "CT scan", "ماموجرام": "mammogram", "ديكسا": "DEXA", "باب سمير": "Pap smear",
    "اندوسكوبي": "endoscopy", "كولونوسكوبي": "colonoscopy", "بيوبسي": "biopsy", "هولتر": "Holter",
    "ستريس تيست": "stress test", "اي اي جي": "EEG", "اي ام جي": "EMG", "اتش اس جي": "HSG",
    "ان تي سكان": "NT scan",
    # ---- تحاليل ----
    "هيموجلوبين": "hemoglobin", "اتش بي": "Hb", "فيريتين": "ferritin", "ايرون": "iron",
    "كالسيوم": "calcium", "بوتاسيوم": "potassium", "صوديوم": "sodium", "ماغنسيوم": "magnesium",
    "جلوكوز": "glucose", "فاستنج": "fasting", "فاستينج": "fasting", "راندوم": "random",
    "يوريك اسيد": "uric acid", "ليفر فانكشن": "liver function", "كيدني فانكشن": "kidney function",
    "ثايرويد بروفايل": "thyroid profile", "ليبيد بروفايل": "lipid profile", "بي اس ايه": "PSA",
    "اتش سي جي": "hCG", "اف اس اتش": "FSH", "ال اتش": "LH", "بروجيستيرون": "progesterone",
    "استرادايول": "estradiol", "برولاكتين": "prolactin", "تستوستيرون": "testosterone",
    "كورتيزول": "cortisol", "هيليكوباكتر": "H. pylori", "بايلوري": "H. pylori", "ستول": "stool",
    "كلتشر": "culture", "يورين كلتشر": "urine culture", "سيمن": "semen", "تروبونين": "troponin",
    "سي كي ام بي": "CK-MB", "اي ان ار": "INR", "بي سي ار": "PCR", "دي دايمر": "D-dimer",
    "اتش سي في": "HCV", "اتش اي في": "HIV", "اتش بي اس اي جي": "HBsAg", "اي اس ار": "ESR",
    "اي ال تي": "ALT", "ايه ال تي": "ALT", "اي اس تي": "AST", "ايه اس تي": "AST",
    "اس بي او تو": "SpO2", "بي ام اي": "BMI", "ايه بي جي": "ABG", "ايزي جي": "ECG", "ايزي شي": "ECG",
    # ---- قلب وصدر وهضمي ----
    "هارت فيلير": "heart failure", "ايسكيميا": "ischemia", "انفاركشن": "infarction",
    "اتريال فيبريليشن": "atrial fibrillation", "ثرومبوسيس": "thrombosis", "انيوريزم": "aneurysm",
    "تاكيكارديا": "tachycardia", "اريثميا": "arrhythmia", "انجاينا": "angina", "ستروك": "stroke",
    "هايبوتنشن": "hypotension", "كوبد": "COPD", "نيمونيا": "pneumonia", "نيومونيا": "pneumonia",
    "برونكايتس": "bronchitis", "كوفيد": "COVID", "جاستريتس": "gastritis", "ريفلكس": "reflux",
    "جيرد": "GERD", "اسيديتي": "acidity", "هارت بيرن": "heartburn", "دسبيبسيا": "dyspepsia",
    "اولسر": "ulcer", "بيبتيك اولسر": "peptic ulcer", "كونستيبيشن": "constipation",
    "هيموروندز": "hemorrhoids", "كولايتس": "colitis", "جاستروانتريتس": "gastroenteritis",
    "كوليسيستايتس": "cholecystitis", "بانكرياتايتس": "pancreatitis", "جالستونز": "gallstones",
    "فاتي ليفر": "fatty liver", "هيباتايتس": "hepatitis", "هيباتايتس سي": "hepatitis C",
    "هيباتايتس بي": "hepatitis B", "سيروسيس": "cirrhosis", "ابنديسايتس": "appendicitis",
    "هيرنيا": "hernia",
    # ---- سكر وهرمونات وكلى ----
    "بري دايابيتس": "pre-diabetes", "انسولين ريزيستانس": "insulin resistance",
    "ميتابوليك سيندروم": "metabolic syndrome", "هيبوجلايسيميا": "hypoglycemia",
    "هايبرجلايسيميا": "hyperglycemia", "كيتواسيدوسيس": "ketoacidosis", "نيوروباثي": "neuropathy",
    "نيفروباثي": "nephropathy", "ريتينوباثي": "retinopathy", "اوبيسيتي": "obesity",
    "ديهيدريشن": "dehydration", "كيدني ستونز": "kidney stones", "سيستايتس": "cystitis",
    "بروستاتايتس": "prostatitis", "بيلونفرايتس": "pyelonephritis", "يو تي اي": "UTI",
    "جويتر": "goiter", "نودول": "nodule", "هايبو ثايرويد": "hypothyroidism",
    "هايبر ثايرويد": "hyperthyroidism",
    # ---- نسا وتوليد ----
    "بي سي او اس": "PCOS", "بوليسيستيك": "polycystic", "فايبرويد": "fibroid", "فايبرويدز": "fibroids",
    "اوفيريان سيست": "ovarian cyst", "سيست": "cyst", "بليفيك": "pelvic", "ايكتوبيك": "ectopic",
    "بري ايكلامبسيا": "preeclampsia", "جيستيشنال دايابيتس": "gestational diabetes",
    "مينوبوز": "menopause", "ديسمينوريا": "dysmenorrhea", "منوراجيا": "menorrhagia",
    "اوفيوليشن": "ovulation", "بي ام اس": "PMS", "بي اي دي": "PID", "سي سكشن": "C-section",
    "هوت فلاشز": "hot flashes",
    # ---- مخ وأعصاب وعظام ----
    "باركنسون": "Parkinson", "الزهايمر": "Alzheimer", "ابيليبسي": "epilepsy", "فيرتيجو": "vertigo",
    "انسومنيا": "insomnia", "انكزايتي": "anxiety", "ديبريشن": "depression",
    "بانيك اتاك": "panic attack", "اوسي دي": "OCD", "بي تي اس دي": "PTSD", "بايبولار": "bipolar",
    "شيزوفرينيا": "schizophrenia", "ستريس": "stress", "ايه دي اتش دي": "ADHD", "سياتيكا": "sciatica",
    "ديسك": "disc", "سبوندايلوسيس": "spondylosis", "كارپال تانل": "carpal tunnel",
    "فايبرومايالجيا": "fibromyalgia", "اوستيوارثرايتس": "osteoarthritis", "ارثرايتس": "arthritis",
    "رومتويد": "rheumatoid", "لوبس": "lupus", "اوستيوبوروسيس": "osteoporosis", "جاوت": "gout",
    "دي في تي": "DVT",
    # ---- جلد وعيون وأنف وودن ----
    "اكني": "acne", "اورتيكاريا": "urticaria", "فيتيليجو": "vitiligo", "دراماتايتس": "dermatitis",
    "اكزيما": "eczema", "سورياسيس": "psoriasis", "سينوسايتس": "sinusitis", "ساينوسايتس": "sinusitis",
    "اوتايتس": "otitis", "فارينجايتس": "pharyngitis", "لارينجايتس": "laryngitis",
    "تونسيلايتس": "tonsillitis", "رينايتس": "rhinitis", "اليرجيك رينايتس": "allergic rhinitis",
    "ادينويد": "adenoids", "كونجنكتيفايتس": "conjunctivitis", "جلوكوما": "glaucoma",
    "كاتاراكت": "cataract",
    # ---- أورام ----
    "كانسر": "cancer", "تيومر": "tumor", "بوليب": "polyp", "ليمفوما": "lymphoma",
    "ليوكيميا": "leukemia", "كيموثيرابي": "chemotherapy", "راديوثيرابي": "radiotherapy",
    "ميتاستاسيس": "metastasis",
    # ---- أعراض بتتقال بالإنجليزي ----
    "بالبيتيشن": "palpitations", "ديسبنيا": "dyspnea", "فاتيج": "fatigue", "راش": "rash",
    "كرامبس": "cramps", "سبازم": "spasm", "نامبنس": "numbness", "تينجلنج": "tingling",
    "بلوتنج": "bloating",
    # ---- فيتامينات ومكملات ----
    "فيتامين سي": "Vitamin C", "فيتامين دال": "Vitamin D", "فيتامين بي كومبلكس": "Vitamin B complex",
    "فيتامين ايه": "Vitamin A", "فيتامين كي": "Vitamin K", "فوليك اسيد": "folic acid",
    "كولاجين": "collagen", "بيوتين": "biotin",
    # ---- كلام إنجليزي شايع في العيادة ----
    "تشيك اب": "check-up", "فولو اب": "follow-up", "ريزلت": "result", "سكان": "scan",
    "سايد افكت": "side effect", "سايد افكتس": "side effects", "اوبريشن": "operation",
    "كانيولا": "cannula", "انفكشن": "infection", "انفلاميشن": "inflammation",
}

# ===== الأدوية: أي جملة فيها اسم دوا اتصلّح بتتعلّم للمراجعة (اسم دوا غلط ممكن يضر) =====
DRUG_DICT = {
    # ---- (من عندك) أدوية ----
    "كونكور": "Concor", "بانادول": "Panadol", "بنادول": "Panadol", "بروفين": "Brufen",
    "فولتارين": "Voltaren", "كتافلام": "Cataflam", "اوجمنتين": "Augmentin", "أوجمنتين": "Augmentin",
    "نيكسيوم": "Nexium", "جلوكوفاج": "Glucophage", "جلوكوفاچ": "Glucophage", "كلاريتين": "Claritin",
    "زيرتك": "Zyrtec", "فنتولين": "Ventolin", "ميتفورمين": "Metformin", "اسبرين": "Aspirin",
    "أسبرين": "Aspirin", "انسولين": "insulin", "إنسولين": "insulin", "كورتيزون": "cortisone",
    "انتيبيوتيك": "antibiotic", "أنتيبيوتيك": "antibiotic", "اتورفاستاتين": "Atorvastatin",
    "املوديبين": "Amlodipine",
    # ---- مسكنات وخافض حرارة ----
    "بروفينه": "Brufen", "ديكلوفيناك": "diclofenac", "بونستان": "Ponstan", "اركوكسيا": "Arcoxia",
    "سيليبركس": "Celebrex", "ترامادول": "tramadol", "ترامال": "Tramal", "باراسيتامول": "paracetamol",
    "ايبوبروفين": "ibuprofen", "نابروكسين": "naproxen",
    # ---- مضادات حيوية ومضادات عدوى ----
    "اموكسيل": "Amoxil", "اوجمانتين": "Augmentin", "زيثروماكس": "Zithromax", "كلاسيد": "Klacid",
    "زينات": "Zinnat", "سوبراكس": "Suprax", "روسيفين": "Rocephin", "ازيثروميسين": "azithromycin",
    "اموكسيسيلين": "amoxicillin", "سيبروفلوكساسين": "ciprofloxacin", "ميترونيدازول": "metronidazole",
    "سيفترياكسون": "ceftriaxone", "بنسلين": "penicillin", "فلاجيل": "Flagyl", "ديفلوكان": "Diflucan",
    "فلوكونازول": "fluconazole", "كانستين": "Canesten", "زوفيراكس": "Zovirax",
    "اسيكلوفير": "acyclovir", "تاميفلو": "Tamiflu",
    # ---- هضمي ----
    "كونترولوك": "Controloc", "انتينال": "Antinal", "بوسكوبان": "Buscopan", "اوميبرازول": "omeprazole",
    "ايزوميبرازول": "esomeprazole", "بانتوبرازول": "pantoprazole", "جافيسكون": "Gaviscon",
    "موتيليوم": "Motilium", "دوسباتالين": "Duspatalin", "دوفالاك": "Duphalac", "لاكتولوز": "lactulose",
    # ---- صدر وحساسية ----
    "اوتريفين": "Otrivin", "بريكانيل": "Bricanyl", "سيريتايد": "Seretide", "بيرودوال": "Berodual",
    "بولميكورت": "Pulmicort", "سنجيولير": "Singulair", "مونتيلوكاست": "montelukast",
    # ---- قلب وضغط وسيولة ----
    "ليبيتور": "Lipitor", "نورفاسك": "Norvasc", "كوزار": "Cozaar", "لازيكس": "Lasix",
    "الداكتون": "Aldactone", "كابوتين": "Capoten", "تنورمين": "Tenormin", "انديرال": "Inderal",
    "ديوفان": "Diovan", "فالسارتان": "valsartan", "اتينولول": "atenolol", "بيسوبرولول": "bisoprolol",
    "كارفيديلول": "carvedilol", "ديجوكسين": "digoxin", "كلكسان": "Clexane", "بلافيكس": "Plavix",
    "وارفارين": "warfarin", "زاريلتو": "Xarelto", "ايليكويس": "Eliquis", "سيمفاستاتين": "simvastatin",
    "روزوفاستاتين": "rosuvastatin", "كريستور": "Crestor",
    # ---- سكر ----
    "اميريل": "Amaryl", "جانوفيا": "Januvia", "جاردينس": "Jardiance", "لانتوس": "Lantus",
    "نوفورابيد": "NovoRapid", "ميكستارد": "Mixtard", "ديابيتون": "Diabeton",
    # ---- هرمونات ونسا ----
    "كلوميد": "Clomid", "بروجينوفا": "Progynova", "دوفاستون": "Duphaston", "يوتروجستان": "Utrogestan",
    "سيتوتك": "Cytotec", "كابرجولين": "cabergoline", "دوستينكس": "Dostinex", "ايلتروكسين": "Eltroxin",
    "ليفوثيروكسين": "levothyroxine", "بريدنيزولون": "prednisolone", "ديكساميثازون": "dexamethasone",
    # ---- أعصاب ونفسي ----
    "سيبراليكس": "Cipralex", "بروزاك": "Prozac", "زولوفت": "Zoloft", "زاناكس": "Xanax",
    "نيورونتين": "Neurontin", "ليريكا": "Lyrica", "ديباكين": "Depakine", "تيجريتول": "Tegretol",
    # ---- أدوية تانية ----
    "بيتادين": "Betadine",
}

MEDICAL_DICT.update(CUSTOM_TERMS)

HOTWORDS_LIST = ["MRI", "CT", "X-ray", "ECG", "Echo", "CBC", "HbA1c", "TSH", "CRP",
                 "hypertension", "diabetes", "insulin", "antibiotic", "cortisone",
                 "Panadol", "Concor", "Augmentin", "Cataflam", "Voltaren", "Brufen",
                 "Metformin", "Aspirin", "Vitamin D", "cholesterol", "thyroid"] + list(CUSTOM_HOTWORDS)
HOTWORDS = None   # متوقفة: القايمة كانت بتخلّي Whisper يكرر الكلمات (X-ray X-ray ...). الاعتماد على الـ prompt والقاموس

WHISPER_PROMPT = ("بقالي كام يوم عندي صداع ووجع في دماغي، وعملت CBC و MRI إمبارح، "
                  "وباخد Panadol بس مفيش فايدة. يعني... أممم، مش عارف.")

_AR = r"\u0600-\u06FF"
_CLS = {"ا": "[اأإآ]", "أ": "[اأإآ]", "إ": "[اأإآ]", "آ": "[اأإآ]",
        "ة": "[هة]", "ه": "[هة]", "ى": "[يى]", "ي": "[يى]"}

def _pat(key):
    body = "".join(r"\s+" if ch == " " else _CLS.get(ch, re.escape(ch)) for ch in key)
    return re.compile(rf"(?<![{_AR}]){body}(?![{_AR}])")

REPLACERS = [(_pat(k), v) for k, v in sorted(MEDICAL_DICT.items(), key=lambda kv: len(kv[0]), reverse=True)]

_LABEL = re.compile(r"(?<![\u0600-\u06FF])(?:المريضة|المريض|الدكتورة|الدكتور|الطبيبة|الطبيب)\s*[:：]\s*")

def clean_text(t):
    """يشيل أي 'المريض:' أو علامات تنصيص زيادة بيكتبها Whisper"""
    t = _LABEL.sub("", t.strip())
    t = t.strip().strip('"“”«»').strip()
    t = re.sub(r"([؟?!])\s*\.", r"\1", t)
    return t.strip()

def apply_dictionary(text):
    text = clean_text(text)
    for p, v in REPLACERS:
        text = p.sub(v, text)
    return text
print("dictionary entries:", len(MEDICAL_DICT))


# ## 6) تنظيف الصوت (خفيف)

# In[33]:


NOISE_FILTERS = {
    "off":    "loudnorm=I=-18:TP=-2",
    "light":  "highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:TP=-2",
    "medium": "highpass=f=100,lowpass=f=7500,afftdn=nf=-30:nr=12,loudnorm=I=-18:TP=-2",
}

def preprocess(src, dst):
    """mono + 16kHz + تنظيف نويز خفيف + توحيد مستوى الصوت"""
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
           "-af", NOISE_FILTERS[NOISE], dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-600:])


# ## 7) التفريغ (Whisper) + تحديد المتكلمين (pyannote)

# In[34]:


import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_asr():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device=DEVICE,
                        compute_type="float16" if DEVICE == "cuda" else "int8")

def read_audio(wav):
    """نقرأ الملف بنفسنا (بدل مكتبة av) ونديه جاهز للموديلات"""
    import soundfile as sf
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def transcribe_words(model, wav):
    audio, _ = read_audio(wav)
    kw = dict(language=LANGUAGE, task="transcribe", initial_prompt=WHISPER_PROMPT,
              word_timestamps=True, vad_filter=True,
              vad_parameters=dict(min_silence_duration_ms=500),
              condition_on_previous_text=False, beam_size=BEAM_SIZE,
              temperature=[0.0, 0.2, 0.4],   # إعادة محاولة بدرجات منخفضة بس (الدرجات العالية بتطلّع كلام عشوائي)
              compression_ratio_threshold=2.0, log_prob_threshold=-1.0, no_speech_threshold=0.6)
    try:
        segments, _ = model.transcribe(audio, hotwords=HOTWORDS, **kw)
        segments = list(segments)
    except TypeError as e:
        if "hotwords" not in str(e):
            raise
        segments, _ = model.transcribe(audio, **kw)      # نسخة قديمة من faster-whisper من غير hotwords
        segments = list(segments)
    words = []
    for s in segments:
        for w_ in (s.words or []):
            words.append({"start": w_.start, "end": w_.end, "text": w_.word})
    return words

def load_diar():
    from pyannote.audio import Pipeline
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    except TypeError:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    if pipe is None:
        raise RuntimeError("لم يتم تحميل pyannote: اتأكدي إنك وافقتي على شروط الموديلين على Hugging Face وإن الـ HF_TOKEN صح.")
    if DEVICE == "cuda":
        pipe.to(torch.device("cuda"))
    return pipe

def diarize(pipe, wav):
    audio, sr = read_audio(wav)
    inp = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}   # من غير ما pyannote يقرا الملف
    kw = {}
    if MIN_SPEAKERS and MAX_SPEAKERS and MIN_SPEAKERS == MAX_SPEAKERS:
        kw["num_speakers"] = MIN_SPEAKERS
    else:
        if MIN_SPEAKERS: kw["min_speakers"] = MIN_SPEAKERS
        if MAX_SPEAKERS: kw["max_speakers"] = MAX_SPEAKERS
    out = pipe(inp, **kw)
    ann = getattr(out, "speaker_diarization", out)        # يشتغل مع نسخ pyannote المختلفة
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]

def assign_speakers(words, turns):
    for w_ in words:
        best, best_ov = None, 0.0
        for s, e, spk in turns:
            ov = min(w_["end"], e) - max(w_["start"], s)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None and turns:                         # كلمة وقعت في فجوة: أقرب متكلم
            mid = (w_["start"] + w_["end"]) / 2
            best = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
        w_["speaker"] = best

def drop_minor_speakers(words):
    """يدمج المتكلمين اللي كلامهم قليل جداً (غالباً غلط من الـ diarization)"""
    t = {}
    for w_ in words:
        t[w_["speaker"]] = t.get(w_["speaker"], 0) + (w_["end"] - w_["start"])
    total = sum(t.values()) or 1
    minor = {s for s, v in t.items() if v < max(MIN_SPEAKER_SECONDS, 0.03 * total)}
    if not minor or len(minor) == len(t):
        return
    for i, w_ in enumerate(words):
        if w_["speaker"] in minor:
            prev = next((words[j]["speaker"] for j in range(i - 1, -1, -1) if words[j]["speaker"] not in minor), None)
            nxt = next((words[j]["speaker"] for j in range(i + 1, len(words)) if words[j]["speaker"] not in minor), None)
            w_["speaker"] = prev or nxt

def group_segments(words, max_gap=2.0):
    segs, cur = [], None
    for w_ in words:
        if cur and cur["speaker"] == w_["speaker"] and (w_["start"] - cur["end"]) < max_gap:
            cur["end"] = w_["end"]; cur["text"] += w_["text"]
        else:
            if cur: segs.append(cur)
            cur = {"speaker": w_["speaker"], "start": w_["start"], "end": w_["end"], "text": w_["text"]}
    if cur: segs.append(cur)
    for s in segs:
        s["text"] = s["text"].strip()
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return segs


# ## 8) الـ LLM: تصحيح النص + تحديد الدكتور والمريض

# In[35]:


CORRECT_SYSTEM = """You are correcting an automatic speech-recognition (ASR) transcript of an Egyptian Arabic doctor-patient consultation.
The ASR often mishears Egyptian dialect words. Using the medical context, fix words that are clearly misheard
(example: "القلم" in a sentence about pain -> "الألم"; "برهسيان" -> "غثيان").
Also: English medical terms, drug names and test names must be written in English letters (كونكور -> Concor, سي بي سي -> CBC).

STRICT RULES:
- Keep the Egyptian dialect exactly as spoken. Do NOT convert to Modern Standard Arabic and do NOT translate Arabic words to English.
- Keep hesitations, repetitions and fillers (يعني، أه، أممم) as they are.
- Do NOT add information, do NOT summarize, do NOT rephrase whole sentences, do NOT invent diagnoses, drug names or doses that were not said.
- Change only the specific words that are clearly wrong. Keep the number of words about the same.
- If you are not sure about a word, leave it unchanged.
Input is a JSON list of {"id": int, "text": str}. Return ONLY a JSON list with the same ids and the corrected "text". No markdown, no commentary."""

ROLE_SYSTEM = """Below is a transcript of a medical consultation in Egyptian Arabic. Speakers are anonymous labels.
Assign each label one role: "Doctor", "Patient", or "Other" (e.g. a companion or nurse).
The doctor asks clinical questions, diagnoses, prescribes and explains treatment.
Exactly one label must be "Doctor" and at least one must be "Patient".
Return ONLY JSON such as {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}."""

def parse_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    i = min([x for x in (s.find("["), s.find("{")) if x >= 0], default=0)
    return json.loads(s[i:])

class ClaudeLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
    def ask(self, system, user, max_tokens=4096):
        r = self.client.messages.create(model=CLAUDE_MODEL, max_tokens=max_tokens, temperature=0,
                                        system=system, messages=[{"role": "user", "content": user}])
        return r.content[0].text

class QwenLLM:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, quantization_config=bnb, device_map="auto")
    def ask(self, system, user, max_tokens=2048):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)     # Qwen3: من غير "تفكير" طويل
        inp = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
        res = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", res, flags=re.S).strip()

def make_llm():
    if LLM_BACKEND == "claude": return ClaudeLLM()
    if LLM_BACKEND == "qwen":   return QwenLLM()
    return None

def safe_text(original, corrected, min_ratio=0.6):
    """حارس: يرفض أي تعديل بيغيّر الجملة كتير (عشان الكلام يفضل بالحرف)"""
    if not isinstance(corrected, str) or not corrected.strip():
        return original
    if abs(len(original.split()) - len(corrected.split())) > 1:
        return original
    return corrected if difflib.SequenceMatcher(None, original, corrected).ratio() >= min_ratio else original

def llm_correct(segs, llm):
    for i in range(0, len(segs), LLM_BATCH):
        chunk = segs[i:i + LLM_BATCH]
        payload = [{"id": j, "text": s["text"]} for j, s in enumerate(chunk)]
        try:
            fixed = parse_json(llm.ask(CORRECT_SYSTEM, json.dumps(payload, ensure_ascii=False)))
            by_id = {f["id"]: f["text"] for f in fixed}
        except Exception as e:
            print(f"  [warn] LLM batch {i} failed, keeping dictionary text: {e}")
            by_id = {}
        for j, s in enumerate(chunk):
            s["text"] = safe_text(s["text"], by_id.get(j, s["text"]))

_Q = re.compile(r"(؟|\?|(?<![\u0600-\u06FF])(إيه|ايه|ليه|امتى|إمتى|فين|إزاي|ازاي|كام|هل|مين|بتحس|حاسس|حاسة)(?![\u0600-\u06FF]))")

def heuristic_roles(segs):
    labels = list(dict.fromkeys(s["speaker"] for s in segs))
    if len(labels) == 1:
        return {labels[0]: "Speaker_1"}
    talk = {l: sum(s["end"] - s["start"] for s in segs if s["speaker"] == l) for l in labels}
    q = {l: sum(len(_Q.findall(s["text"])) for s in segs if s["speaker"] == l) for l in labels}
    doctor = max(labels, key=lambda l: (q[l], -labels.index(l)))
    rest = sorted([l for l in labels if l != doctor], key=lambda l: -talk[l])
    roles = {doctor: "Doctor", rest[0]: "Patient"}
    for k, l in enumerate(rest[1:], 1):
        roles[l] = f"Other_{k}"
    return roles

def identify_roles(segs, llm):
    labels = list(dict.fromkeys(s["speaker"] for s in segs))
    roles = heuristic_roles(segs)
    if llm and len(labels) >= 2:
        sample = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segs[:50])
        try:
            r = parse_json(llm.ask(ROLE_SYSTEM, sample, 300))
            ok = (set(r) == set(labels) and list(r.values()).count("Doctor") == 1
                  and "Patient" in r.values() and set(r.values()) <= {"Doctor", "Patient", "Other"})
            if ok:
                roles, k = {}, 0
                for l in labels:
                    if r[l] == "Other":
                        k += 1; roles[l] = f"Other_{k}"
                    else:
                        roles[l] = r[l]
            else:
                print("  [warn] LLM roles invalid, using heuristic")
        except Exception as e:
            print(f"  [warn] role detection failed, using heuristic: {e}")
    return roles


# ## 9) تشغيل كل حاجة

# In[36]:


AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".mp4", ".webm"}

SPLIT_SYSTEM = """You receive the words of a doctor-patient consultation in Egyptian Arabic, in order, each with an index (index:word).
Speaker changes were lost. Return the speaker turns as a JSON list of {"start": <index of the first word of the turn>, "speaker": "Doctor" | "Patient" | "Other"}.
Rules:
- The doctor asks clinical questions, explains, orders tests and plans treatment. The patient describes symptoms, complaints and history, and answers the doctor.
- A question usually ends a doctor turn and the answer that follows is the patient's. Split at every change of speaker, even in the middle of a sentence.
- Use "Other" only for a clearly different third person (companion, nurse).
- The first item must start at the first index you were given. Return ONLY the JSON list, no commentary."""

HINT_NOTE = """
Tokens like <V1> or <V2> mark where an audio voice detector believes a different voice starts. They can be wrong or missing when the voices are similar,
so use them only as a weak hint. The meaning of the sentences decides who is speaking."""

def split_turns_by_text(words, llm, chunk=220, hints=False):
    system = SPLIT_SYSTEM + (HINT_NOTE if hints else "")
    labels, prev, vname = [None] * len(words), None, {}
    for a in range(0, len(words), chunk):
        part = words[a:a + chunk]
        toks, last = [], None
        for i, w_ in enumerate(part):
            tok = f"{a + i}:{w_['text'].strip()}"
            if hints and w_.get("speaker") != last:
                last = w_.get("speaker")
                tok = f"<{vname.setdefault(last, 'V' + str(len(vname) + 1))}> " + tok
            toks.append(tok)
        hint = (f"The speaker of the word just before index {a} was: {prev}.\n" if prev
                else "The consultation starts here (the doctor usually speaks first).\n")
        starts = {}
        try:
            for t in parse_json(llm.ask(system, hint + " ".join(toks), 900)):
                i = int(t["start"])
                if a <= i < a + len(part) and t.get("speaker") in ("Doctor", "Patient", "Other"):
                    starts[i] = t["speaker"]
        except Exception as e:
            print(f"  [warn] turn splitting failed for words {a}-{a + len(part)}: {e}")
        cur = starts.get(a, prev or "Doctor")
        for i in range(len(part)):
            cur = starts.get(a + i, cur)
            labels[a + i] = cur
        prev = cur
    return labels

def run_all():
    mode = SPEAKER_MODE
    by_meaning = mode in ("text", "hybrid")
    use_audio = mode in ("audio", "hybrid")
    if by_meaning and LLM_BACKEND == "none":
        raise SystemExit("الوضع ده محتاج LLM: غيّري LLM_BACKEND لـ 'claude' أو 'qwen' في خلية الإعدادات.")
    files = sorted(p for p in Path(RECORDS_DIR).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"مفيش ريكوردات في {RECORDS_DIR}. ارفعيها من خلية الرفع الأول.")
    os.makedirs(TMP_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

    print("loading Whisper" + (" + pyannote" if use_audio else "") + " ...")
    asr = load_asr()
    diar = load_diar() if use_audio else None
    staged = {}
    for f in files:
        print(f"\n==> {f.name}")
        wav = str(Path(TMP_DIR) / (f.stem + ".wav"))
        preprocess(str(f), wav)
        print("  transcribing ...")
        words = transcribe_words(asr, wav)
        if not words:
            print("  [warn] no speech detected"); continue
        if use_audio:
            print("  diarizing ...")
            assign_speakers(words, diarize(diar, wav))
            drop_minor_speakers(words)
        else:
            for w_ in words: w_["speaker"] = "Doctor"
        staged[f] = words
        os.remove(wav)
    del asr, diar
    free_gpu()

    llm = None
    if LLM_BACKEND != "none":
        print(f"\nloading LLM backend: {LLM_BACKEND} ...")
        llm = make_llm()

    for f, words in staged.items():
        print(f"\n==> post-processing {f.name}")
        if by_meaning:
            print("  deciding who speaks (meaning" + (" + voice hints" if use_audio else "") + ") ...")
            for w_, lab in zip(words, split_turns_by_text(words, llm, hints=use_audio)):
                w_["speaker"] = lab
        segs = group_segments(words)
        for s in segs:
            s["text_raw"] = s["text"]
            s["text"] = apply_dictionary(s["text"])
        if llm:
            llm_correct(segs, llm)
        if by_meaning:
            roles = {l: l for l in dict.fromkeys(s["speaker"] for s in segs)}
        else:
            roles = identify_roles(segs, llm)
            for s in segs:
                s["speaker"] = roles.get(s["speaker"], s["speaker"])
        talk = {}
        for s in segs:
            talk[s["speaker"]] = round(talk.get(s["speaker"], 0) + s["end"] - s["start"], 1)
        out = {"file": f.name,
               "speaker_mode": mode,
               "num_speakers": len(roles),
               "speaker_roles": roles,
               "talk_time_seconds": talk,
               "segments": [{"speaker": s["speaker"], "start": s["start"], "end": s["end"],
                             "text": s["text"], "text_raw": s["text_raw"]} for s in segs]}
        p = Path(OUT_DIR) / f"{f.stem}.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  saved ->", p)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\nDONE")

run_all()


# ## 10) شوفي النتيجة ونزّليها

# In[37]:


import json, glob
for p in sorted(glob.glob(OUT_DIR + "/*.json")):
    d = json.load(open(p, encoding="utf-8"))
    print("=" * 70)
    print(d["file"], "| speakers:", d["num_speakers"], "|", d["speaker_roles"], "|", d["talk_time_seconds"])
    for s in d["segments"][:25]:
        print(f'[{s["start"]:>7}s] {s["speaker"]:9}: {s["text"]}')


# In[29]:


import shutil
shutil.make_archive("/kaggle/working/results", "zip", OUT_DIR)
print("saved /kaggle/working/results.zip  (هتلاقيه في تاب Output على اليمين بعد Save Version، أو من File panel)")


# In[ ]:




