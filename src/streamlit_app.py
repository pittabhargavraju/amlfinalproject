import streamlit as st
from ranker import rank_resumes, bias_check
from explanation import make_predict_fn, explainer
from utils import load_resume_text, extract_skills
import tempfile
import os

st.set_page_config(page_title="FairHire AI", layout="wide")

st.title("FairHire AI — Resume Ranking, Skills & Bias-Free Recommendations")

st.write("""
Upload resumes and a job description to see:
- Hybrid Matching Score (TF-IDF + BERT)
- Extracted Skills
- Missing Skills
- Personalized Improvement Suggestions
- Bias Check (Before/After Masking)
- LIME Explainability for Transparency
""")

jd = st.text_area("Paste Job Description Here", height=160)

uploads = st.file_uploader("Upload Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

if st.button("Run Analysis"):
    if jd and uploads:
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        for file in uploads:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(path)

        results = rank_resumes(file_paths, jd)

        st.subheader("📊 Results")

        for res in results:
            st.markdown(f"## 📄 {os.path.basename(res['resume'])}")
            st.markdown(f"### 🎯 Final Hybrid Score: **{res['final_score']} / 100**")

            # Extracted skills
            st.markdown("### 🧠 Extracted Skills:")
            st.info(", ".join(res["recommendations"]["resume_skills"]) or "No skills detected.")

            # Missing skills
            st.markdown("### ❗ Missing Skills:")
            st.warning(", ".join(res["recommendations"]["missing_skills"]) or "None")

            # Suggestions
            st.markdown("### 💡 Suggestions:")
            for s in res["recommendations"]["suggestions"]:
                st.write(f"- {s}")

            # Bias Check
            st.markdown("### ⚖️ Bias Check:")
            bc = bias_check(res["text"], jd)
            st.write(f"Base Score: **{bc['base_score']:.4f}**")
            st.write(f"Masked Score: **{bc['masked_score']:.4f}**")
            st.write(f"Score Difference: **{bc['change']:.4f}**")

            # LIME Explainability
            st.markdown("### 🔍 LIME Explainability")
            predict_fn = make_predict_fn(jd)
            explanation = explainer.explain_instance(
                res["text"],
                predict_fn,
                num_features=5
            )
            st.json(explanation.as_list())

            st.markdown("---")
