import pdfplumber
import docx
import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLP + BERT Model
nlp = spacy.load("en_core_web_sm")
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Master Skill List
SKILL_LIST = [
    "python","sql","docker","kubernetes","pytorch","tensorflow",
    "nlp","machine learning","deep learning","pandas","numpy",
    "scikit-learn","git","aws","azure","spark","hadoop","excel",
    "linux","communication","leadership","teamwork","data analysis",
    "java","c++","javascript","react","node"
]

# ======================================================================
# STEP 2: CLEANING / PREPROCESSING
# ======================================================================
def clean_text(txt):
    txt = re.sub(r"\S+@\S+", " ", txt)                       # remove emails
    txt = re.sub(r"\b\d{10,}\b", " ", txt)                   # remove phone numbers
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)              # remove URLs
    txt = re.sub(r"[^a-zA-Z0-9.,;:()\s]", " ", txt)          # remove symbols
    txt = re.sub(r"\s+", " ", txt).strip().lower()           # collapse spaces + lowercase
    return txt

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return " ".join(p.text for p in doc.paragraphs)

def load_resume_text(path):
    ext = path.split(".")[-1].lower()
    if ext == "pdf":
        raw = extract_text_from_pdf(path)
    elif ext == "docx":
        raw = extract_text_from_docx(path)
    else:
        raw = open(path, "r", encoding="utf-8").read()
    return clean_text(raw)

# ======================================================================
# STEP 3: SKILL EXTRACTION (TF-IDF + BERT Semantic)
# ======================================================================

# 1️⃣ TF-IDF keyword-based skill extraction
def extract_skills_tfidf(text):
    vectorizer = TfidfVectorizer(vocabulary=SKILL_LIST)
    matrix = vectorizer.fit_transform([text])
    scores = matrix.toarray()[0]

    extracted = [
        SKILL_LIST[i] for i, score in enumerate(scores) if score > 0
    ]
    return extracted

# 2️⃣ BERT semantic skill extraction
def extract_skills_bert(text):
    text_emb = EMB_MODEL.encode([text])[0]
    skill_embs = EMB_MODEL.encode(SKILL_LIST)

    extracted = []
    for skill, emb in zip(SKILL_LIST, skill_embs):
        sim = np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb))
        if sim > 0.55:  # semantic threshold
            extracted.append(skill)
    return extracted

# 3️⃣ Combined hybrid skill extraction
def extract_skills(text):
    tfidf_skills = extract_skills_tfidf(text)
    bert_skills = extract_skills_bert(text)
    skills = list(set(tfidf_skills + bert_skills))
    return skills

# ======================================================================
# STEP 4: Embedding Creation (TF-IDF + BERT)
# ======================================================================

# TF-IDF vectorizer for sparse embedding
TFIDF_VECTORIZER = TfidfVectorizer()

def tfidf_embed(text1, text2):
    """
    Returns TF-IDF vectors for resume and job description.
    Needed for hybrid scoring.
    """
    matrix = TFIDF_VECTORIZER.fit_transform([text1, text2])
    resume_vec = matrix.toarray()[0]
    jd_vec = matrix.toarray()[1]
    return resume_vec, jd_vec

def tfidf_similarity(text1, text2):
    """
    Cosine similarity using TF-IDF sparse vectors.
    """
    v1, v2 = tfidf_embed(text1, text2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))


# BERT dense embedding
def embed_text(text):
    return EMB_MODEL.encode([text])[0]

def bert_similarity(resume_text, jd_text):
    v1 = embed_text(resume_text)
    v2 = embed_text(jd_text)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ======================================================================
# Recommendations based on missing skills (stays same)
# ======================================================================
def recommend_actions(resume_text, jd_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    missing = [s for s in jd_skills if s not in resume_skills]

    suggestions = [f"Add or improve: {skill}" for skill in missing]

    return {
        "resume_skills": resume_skills,
        "missing_skills": missing,
        "suggestions": suggestions
    }
