from lime.lime_text import LimeTextExplainer
from utils import EMB_MODEL, embed_text
import numpy as np

explainer = LimeTextExplainer(class_names=["match"])

def make_predict_fn(jd_text):
    """
    Creates a prediction function for LIME using cosine similarity.
    """
    jd_vec = embed_text(jd_text)

    def predict(texts):
        text_embeddings = EMB_MODEL.encode(texts, convert_to_numpy=True)
        sims = np.dot(text_embeddings, jd_vec) / (
            np.linalg.norm(jd_vec) * np.linalg.norm(text_embeddings, axis=1)
        )
        probs = np.vstack([sims, 1 - sims]).T
        return probs

    return predict
