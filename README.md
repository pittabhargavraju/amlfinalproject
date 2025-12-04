# FairHire AI: Bias-Free Resume Ranking Using Hybrid Deep Learning

FairHire AI is an intelligent resume-ranking system designed to match resumes with job descriptions in a fair, transparent, and accurate manner. It leverages a hybrid deep learning approach combining state-of-the-art Natural Language Processing (NLP) techniques.

This project was developed as part of an academic assignment focusing on advanced Deep Learning applications in NLP.

---

## Project Overview

Traditional Applicant Tracking Systems (ATS) often suffer from reliance on simple keyword matching, leading to semantic misunderstandings and potential ranking bias. FairHire AI addresses these challenges through a robust hybrid semantic pipeline.

### Key Challenges Addressed:

- Keyword-based systems fail to understand **semantic meaning**.
- Bias may influence rankings due to demographic information.
- The need for fast, automated screening of large resume volumes.

### Core Solutions:

1.  **Hybrid Scoring:** Combines deep semantic understanding (SBERT) with keyword relevance (TF-IDF).
2.  **Bias Mitigation:** Masks sensitive terms to ensure fairness in ranking.
3.  **Explainability (XAI):** Uses LIME to provide transparency for the ranking score.
4.  **Interactive Interface:** A Streamlit Web Application for practical use.

---

## Methodology (Eight-Stage Pipeline)

FairHire AI utilizes an eight-stage pipeline for processing and ranking resumes:

1.  **Resume Input:** Accepts and extracts raw text from PDF, DOCX, and TXT files.
2.  **Preprocessing:** Cleans text by removing emails, phone numbers, URLs, special characters, and converting to lowercase.
3.  **Skill Extraction (Hybrid):** Uses parallel TF-IDF (explicit keywords) and BERT (semantic context) techniques.
4.  **Embedding Creation:** Generates both a sparse **TF-IDF vector** and a dense **BERT/SBERT embedding**.
5.  **Similarity Calculation:** Computes **Cosine Similarity** for both embedding types against the job description.
6.  **Hybrid Scoring:** Calculates the final score using a weighted average:
    $$
    \text{Final Score} = 0.5 \times \text{BERT Semantic Similarity} + 0.5 \times \text{TF-IDF Similarity}
    $$
7.  **Bias Removal:** Masks sensitive demographic indicators, followed by a re-scoring step to assess bias impact.
8.  **Ranking & Feedback:** Normalizes scores (0-100), ranks resumes, and provides extracted skills, missing skills, and suggested improvements.

---

## Tech Stack

| Category                      | Tools & Libraries                                                          |
| :---------------------------- | :------------------------------------------------------------------------- |
| **Core NLP & ML**             | Sentence-BERT (SBERT), TF-IDF Vectorizer, Cosine Similarity, spaCy NLP     |
| **Fairness & Explainability** | Bias Masking Rules, LIME (Local Interpretable Model-Agnostic Explanations) |
| **Frontend**                  | Streamlit Web Application                                                  |
| **Programming Language**      | Python 3                                                                   |

---

## How to Run the System Locally

Follow these steps to set up and run the FairHire AI system on your local machine:

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run src/streamlit_app.py
    ```
    Upload your resumes and paste the job description into the web interface.

---

## Features

- **Hybrid Scoring:** Combines semantic understanding and keyword matching for superior relevance.
- **Automatic Skill Extraction:** Comprehensive extraction via two parallel NLP methods.
- **Fairness:** Active bias removal mechanisms to mitigate demographic influence.
- **Transparency:** Scoring explainability provided by LIME.
- **Actionable Feedback:** Recommendations for missing skills to improve candidacy.

---

## Evaluation

FairHire AI was rigorously evaluated against several models (TF-IDF, Word2Vec, Doc2Vec, standard BERT/RoBERTa) and achieved the **highest semantic matching performance** due to its unique hybrid scoring methodology.

---

## ⏭ Future Work

- Multilingual resume support.
- Integration with enterprise HR dashboards for a complete ATS solution.
- Retrieval-Augmented Generation (RAG)-based scoring for enhanced contextual relevance.
- OCR support for scanned PDF documents.
