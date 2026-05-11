
# ============================================================
# app.py  –  Medical Image Caption Retrieval Web Interface
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re, string, os

import nltk
from nltk.corpus import stopwords
for r in ["stopwords", "punkt"]:
    nltk.download(r, quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CAPTION_FILE = "image_caption.txt"
GLOVE_PATH   = "glove.6B.100d.txt"
EMBED_DIM    = 100
STOP_WORDS   = set(stopwords.words("english"))

# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------
def preprocess_medical(text: str) -> str:
    """Lowercase, remove punctuation/numbers, strip stop words."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

# ------------------------------------------------------------------
# Cached data loading (runs once, then Streamlit caches the result)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CAPTION_FILE, sep="\t")
    df.columns = df.columns.str.strip()
    df["clean"] = df["caption"].astype(str).apply(preprocess_medical)
    return df

@st.cache_resource
def build_tfidf(df):
    vec = TfidfVectorizer(max_features=5000)
    tdm = vec.fit_transform(df["clean"])
    return vec, tdm

@st.cache_resource
def load_glove_embeddings(df):
    """Load GloVe and pre-compute caption embeddings (cached)."""
    if not os.path.exists(GLOVE_PATH):
        return None, None
    embeddings = {}
    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            embeddings[parts[0]] = np.array(parts[1:], dtype="float32")
    def avg_embed(text):
        vecs = [embeddings[w] for w in text.split() if w in embeddings]
        return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM)
    cap_embs = np.vstack([avg_embed(t) for t in df["clean"]])
    return embeddings, cap_embs

# ------------------------------------------------------------------
# Retrieval helpers
# ------------------------------------------------------------------
def tfidf_retrieve(query, df, vec, tdm, top_k):
    q_vec = vec.transform([preprocess_medical(query)])
    sims  = cosine_similarity(q_vec, tdm).flatten()
    idx   = np.argsort(sims)[::-1][:top_k]
    out   = df.iloc[idx][["ID", "caption"]].copy()
    out["similarity"] = sims[idx].round(4)
    return out.reset_index(drop=True)

def glove_retrieve(query, df, embed_dict, cap_embs, top_k):
    def avg_embed(text):
        vecs = [embed_dict[w] for w in text.split() if w in embed_dict]
        return np.mean(vecs, axis=0) if vecs else np.zeros(EMBED_DIM)
    q_vec = avg_embed(preprocess_medical(query)).reshape(1, -1)
    sims  = cosine_similarity(q_vec, cap_embs).flatten()
    idx   = np.argsort(sims)[::-1][:top_k]
    out   = df.iloc[idx][["ID", "caption"]].copy()
    out["similarity"] = sims[idx].round(4)
    return out.reset_index(drop=True)

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Medical Caption Retrieval", layout="wide")
st.title("🏥 Medical Image Caption Retrieval")
st.markdown(
    "Search 1,000 medical image captions using **TF-IDF** and/or **GloVe** embeddings."
)

# Load resources
df       = load_data()
vec, tdm = build_tfidf(df)
glove_dict, cap_embs = load_glove_embeddings(df)

# Sidebar controls
st.sidebar.header("Search Settings")
top_k  = st.sidebar.slider("Number of results (K)", min_value=5, max_value=10, value=10)
method = st.sidebar.selectbox("Retrieval method",
                               ["TF-IDF", "GloVe (if available)", "Both"])

# Pre-defined queries + custom input
preset_queries = [
    "angiographic image shows normal coronary artery",
    "CT scan demonstrating pulmonary embolism",
    "MRI showing brain tumor mass",
    "ultrasound of gallbladder with stones",
    "(Custom — type below)",
]
selected = st.selectbox("Choose a preset query or enter your own:", preset_queries)

if selected == "(Custom — type below)":
    query = st.text_input("Enter your query:", "")
else:
    query = selected
    st.text_input("Active query:", value=query, disabled=True)

if st.button("🔍  Search") and query.strip():
    st.markdown(f"**Query:** `{query}`")

    if method in ["TF-IDF", "Both"]:
        st.subheader("TF-IDF Results")
        res_tfidf = tfidf_retrieve(query, df, vec, tdm, top_k)
        st.dataframe(res_tfidf, use_container_width=True)

    if method in ["GloVe (if available)", "Both"]:
        st.subheader("GloVe Embedding Results")
        if glove_dict is None:
            st.warning(f"GloVe file not found at `{GLOVE_PATH}`.  "
                       "Download `glove.6B.100d.txt` and place it in the same folder.")
        else:
            res_glove = glove_retrieve(query, df, glove_dict, cap_embs, top_k)
            st.dataframe(res_glove, use_container_width=True)
