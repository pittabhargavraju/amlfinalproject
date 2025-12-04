from utils import (
    load_resume_text,
    recommend_actions,
    bert_similarity,
    tfidf_similarity,
    clean_text
)
import numpy as np
import re

# ===================================================================
# STEP 5: SIMILARITY CALCULATION (BERT + TF-IDF)
# ===================================================================
def compute_similarity(resume_text, jd_text):
    """
    Computes both BERT similarity and TF-IDF similarity.
    """
    bert_score = bert_similarity(resume_text, jd_text)
    tfidf_score = tfidf_similarity(resume_text, jd_text)
    return bert_score, tfidf_score


# ===================================================================
# STEP 6: HYBRID SCORING
# Final Score = 0.5 * BERT + 0.5 * TF-IDF
# ===================================================================
def hybrid_score(resume_text, jd_text):
    bert_score, tfidf_score = compute_similarity(resume_text, jd_text)
    final = 0.5 * bert_score + 0.5 * tfidf_score
    return final


# ===================================================================
# STEP 7: BIAS REMOVAL
# Mask sensitive words + recalc hybrid score
# ===================================================================
SENSITIVE_WORDS = [

    # gender
    "male", "female", "man", "woman", "she", "he", "him", "her",

    # religion
    "hindu", "muslim", "christian", "sikh", "buddhist", "jewish",

    # India cities
    "delhi", "mumbai", "bangalore", "chennai", "hyderabad",
    "kolkata", "pune", "ahmedabad", "surat",

    # USA major cities
    "new york", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "san francisco", "columbus", "fort worth",
    "indianapolis", "seattle", "denver", "oklahoma city", "nashville",
    "boston", "el paso", "portland", "detroit", "memphis",
    "baltimore", "las vegas", "milwaukee", "albuquerque",
    "tucson", "fresno", "sacramento", "kansas city", "mesa",
    "atlanta", "miami", "washington",

    # USA regions
    "east coast", "west coast", "midwest", "south", "northeast", "southwest",

    # nationalities / countries
    "india", "usa", "united states", "canada", "china", "japan", "korea",

    # socio-economic markers (colleges)
    "iit", "nit", "iiit", "stanford", "harvard", "mit",

    # age markers
    "years old", "age", "born"
]


def mask_sensitive_words(text):
    """
    Masks sensitive words to reduce bias.
    """
    masked = text
    for word in SENSITIVE_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        masked = re.sub(pattern, "[MASKED]", masked, flags=re.I)
    return masked


def bias_check(text, jd_text):
    """
    Returns base score, masked score, and score difference.
    """
    base = hybrid_score(text, jd_text)

    masked_text = mask_sensitive_words(text)
    new_score = hybrid_score(masked_text, jd_text)

    return {
        "base_score": base,
        "masked_score": new_score,
        "change": new_score - base
    }


# ===================================================================
# STEP 8: RANKING RESUMES (Normalized 0–100)
# ===================================================================
def rank_resumes(resume_files, jd_text):
    """
    Ranks resumes by hybrid score after full processing.
    """
    jd_text = clean_text(jd_text)
    results = []

    for path in resume_files:
        text = load_resume_text(path)
        score = hybrid_score(text, jd_text)
        recs = recommend_actions(text, jd_text)

        results.append({
            "resume": path,
            "text": text,
            "score": score,
            "recommendations": recs
        })

    # Normalize scores 0–100
    scores = np.array([r["score"] for r in results])
    normalized = 100 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    for i, r in enumerate(results):
        r["final_score"] = float(round(normalized[i], 2))

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
