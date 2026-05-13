
# ============================================================
# app.py - Medical Image Caption Retrieval Web Interface
# Run with: streamlit run app.py
# ============================================================

from pathlib import Path
import re
import string

import streamlit as st
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CAPTION_FILE = Path("image_caption.txt")
GLOVE_PATH = Path("glove.6B.100d.txt")
EMBED_DIM = 100

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def preprocess_medical(text: str) -> str:
    """
    Lightly clean medical caption text:
    - lowercase
    - remove punctuation and numbers
    - remove stop words
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = [
        token
        for token in text.split()
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)


# ------------------------------------------------------------------
# Cached data loading
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load and preprocess the caption dataset."""

    if not CAPTION_FILE.exists():
        st.error(f"Caption file not found: {CAPTION_FILE}")
        st.stop()

    df = pd.read_csv(CAPTION_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    required_columns = {"ID", "caption"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    df["clean"] = df["caption"].apply(preprocess_medical)

    return df


@st.cache_resource
def build_tfidf(clean_text):
    """Build TF-IDF vectorizer and document-term matrix."""

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(clean_text)

    return vectorizer, tfidf_matrix


@st.cache_resource
def load_glove_embeddings(clean_text):
    """
    Load GloVe embeddings and pre-compute caption embeddings.
    Returns None values if GloVe file is unavailable.
    """

    if not GLOVE_PATH.exists():
        return None, None

    embeddings = {}

    with open(GLOVE_PATH, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector

    caption_embeddings = np.vstack([
        average_embedding(text, embeddings)
        for text in clean_text
    ])

    return embeddings, caption_embeddings


def average_embedding(text, embedding_dict):
    """Convert text into an average GloVe embedding vector."""

    vectors = [
        embedding_dict[word]
        for word in text.split()
        if word in embedding_dict
    ]

    if not vectors:
        return np.zeros(EMBED_DIM)

    return np.mean(vectors, axis=0)


# ------------------------------------------------------------------
# Retrieval helpers
# ------------------------------------------------------------------

def tfidf_retrieve(query, df, vectorizer, tfidf_matrix, top_k=10):
    """Return top-K captions using TF-IDF cosine similarity."""

    query_clean = preprocess_medical(query)

    if not query_clean:
        return pd.DataFrame(columns=["ID", "caption", "similarity"])

    query_vector = vectorizer.transform([query_clean])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_k = min(top_k, len(df))
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = df.iloc[top_indices][["ID", "caption"]].copy()
    results["similarity"] = similarity_scores[top_indices].round(4)

    return results.reset_index(drop=True)


def glove_retrieve(query, df, embedding_dict, caption_embeddings, top_k=10):
    """Return top-K captions using GloVe average embedding similarity."""

    query_clean = preprocess_medical(query)
    query_vector = average_embedding(query_clean, embedding_dict)

    if np.all(query_vector == 0):
        return pd.DataFrame(columns=["ID", "caption", "similarity"])

    similarity_scores = cosine_similarity(
        query_vector.reshape(1, -1),
        caption_embeddings
    ).flatten()

    top_k = min(top_k, len(df))
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = df.iloc[top_indices][["ID", "caption"]].copy()
    results["similarity"] = similarity_scores[top_indices].round(4)

    return results.reset_index(drop=True)


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Caption Retrieval",
    layout="wide"
)

st.title("Medical Image Caption Retrieval")

st.markdown(
    """
    Search medical image captions using **TF-IDF** and optional **GloVe**
    word embeddings.
    """
)


# Load resources
df = load_data()
tfidf_vectorizer, tfidf_matrix = build_tfidf(df["clean"])
glove_dict, caption_embeddings = load_glove_embeddings(df["clean"])


# Sidebar controls
st.sidebar.header("Search Settings")

top_k = st.sidebar.slider(
    "Number of results",
    min_value=5,
    max_value=10,
    value=10
)

method = st.sidebar.selectbox(
    "Retrieval method",
    ["TF-IDF", "GloVe", "Both"]
)


# Query input
preset_queries = [
    "angiographic image shows normal coronary artery",
    "CT scan demonstrating pulmonary embolism",
    "MRI showing brain tumor mass",
    "ultrasound of gallbladder with stones",
    "Custom query"
]

selected_query = st.selectbox(
    "Choose a preset query or enter your own:",
    preset_queries
)

if selected_query == "Custom query":
    query = st.text_input("Enter your query:")
else:
    query = selected_query
    st.text_input("Active query:", value=query, disabled=True)


# Search button
if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query before searching.")
        st.stop()

    st.markdown(f"**Query:** `{query}`")

    if method in ["TF-IDF", "Both"]:
        st.subheader("TF-IDF Results")

        tfidf_results = tfidf_retrieve(
            query,
            df,
            tfidf_vectorizer,
            tfidf_matrix,
            top_k
        )

        if tfidf_results.empty:
            st.warning("No TF-IDF results found for this query.")
        else:
            st.dataframe(tfidf_results, use_container_width=True)

    if method in ["GloVe", "Both"]:
        st.subheader("GloVe Embedding Results")

        if glove_dict is None or caption_embeddings is None:
            st.warning(
                f"GloVe file not found at `{GLOVE_PATH}`. "
                "Place `glove.6B.100d.txt` in the same folder as app.py "
                "to enable embedding retrieval."
            )
        else:
            glove_results = glove_retrieve(
                query,
                df,
                glove_dict,
                caption_embeddings,
                top_k
            )

            if glove_results.empty:
                st.warning(
                    "No GloVe results found. The query may not contain words "
                    "covered by the GloVe vocabulary."
                )
            else:
                st.dataframe(glove_results, use_container_width=True)
