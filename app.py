import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Load dataset  - dataset from https://github.com/mauropelucchi/tedx_dataset?utm_source=chatgpt.com
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ted_talks.csv")
    df["description"] = df["description"].fillna("")
    df["presenterDisplayName"] = df["presenterDisplayName"].fillna("")
    return df

df = load_data()

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

# -----------------------------
# Load precomputed embeddings
# -----------------------------
@st.cache_data
def load_embeddings():
    return np.load("ted_embeddings.npy")

embeddings = load_embeddings()

# -----------------------------
# Search talks by keyword/topic
# -----------------------------
def search_talks(topic):
    topic = topic.lower()
    mask = (
        df["slug"].str.lower().str.contains(rf"\b{re.escape(topic)}\b", na=False)
        | df["description"].str.lower().str.contains(rf"\b{re.escape(topic)}\b", na=False)
        | df["socialDescription"].str.lower().str.contains(rf"\b{re.escape(topic)}\b", na=False)
        | df["presenterDisplayName"].str.lower().str.contains(rf"\b{re.escape(topic)}\b", na=False)
    )
    return df[mask]

# -----------------------------
# Recommend similar talks (cosine similarity)
# -----------------------------
def recommend_similar_talks(input_text, top_k=5):
    input_embedding = model.encode([input_text], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    top_results = cos_scores.topk(top_k)
    recommended_indices = top_results.indices.tolist()
    return df.iloc[recommended_indices]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎤 TED Talks Search & Recommendation")
st.write("Search TED Talks by topic or get recommendations based on a talk description.")

# Sidebar filters
st.sidebar.header("⚙️ Filters")
sort_by = st.sidebar.selectbox("Sort by:", ["Relevance", "Newest", "Oldest"])
min_duration, max_duration = st.sidebar.slider(
    "Filter by Duration (seconds)", 0, int(df["duration"].max()), (0, int(df["duration"].max()))
)

# Tabs: Search or Recommend
tab = st.radio("Choose mode:", ["Search Talks", "Recommend Talks"])

# -----------------------------
# Search Mode
# -----------------------------
if tab == "Search Talks":
    topic = st.text_input("🔍 Enter a topic (e.g., AI, motivation, learning English):")
    if topic:
        results = search_talks(topic)
        results = results[(results["duration"] >= min_duration) & (results["duration"] <= max_duration)]

        # Sorting
        if sort_by == "Newest":
            results = results.sort_values("publishedAt", ascending=False)
        elif sort_by == "Oldest":
            results = results.sort_values("publishedAt", ascending=True)

        if results.empty:
            st.warning("No talks found for this topic.")
        else:
            st.success(f"✅ Found {len(results)} talks related to **{topic}**")
            for _, row in results.head(5).iterrows():
                talk_url = f"https://www.ted.com/talks/{row['slug']}"
                st.markdown(f"### 🎬 [{row['slug'].replace('-', ' ').title()}]({talk_url})")
                st.write(f"**📝 Description:** {row['description']}")
                st.write(f"**👤 Presenter:** {row['presenterDisplayName']}")
                st.write(f"**⏱️ Duration:** {row['duration']} seconds")
                st.write(f"**📅 Published at:** {row['publishedAt']}")
                st.markdown(f"👉 [Watch Talk Here]({talk_url})", unsafe_allow_html=True)
                st.markdown("---")

# -----------------------------
# Recommendation Mode
# -----------------------------
elif tab == "Recommend Talks":
    input_text = st.text_area("📝 Paste a TED talk description or topic for recommendations:")
    top_k = st.slider("Number of recommendations:", 1, 10, 5)
    if input_text:
        recommended = recommend_similar_talks(input_text, top_k)
        st.success(f"🎯 Top {top_k} recommended talks:")
        for _, row in recommended.iterrows():
            talk_url = f"https://www.ted.com/talks/{row['slug']}"
            st.markdown(f"### 🎬 [{row['slug'].replace('-', ' ').title()}]({talk_url})")
            st.write(f"**📝 Description:** {row['description']}")
            st.write(f"**👤 Presenter:** {row['presenterDisplayName']}")
            st.write(f"**⏱️ Duration:** {row['duration']} seconds")
            st.write(f"**📅 Published at:** {row['publishedAt']}")
            st.markdown(f"👉 [Watch Talk Here]({talk_url})", unsafe_allow_html=True)
            st.markdown("---")
#TEDx Dataset (Updated March 2025)

# This dataset from GitHub includes metadata on over 7,900 TEDx talks, and was last updated as recently as March 9, 2025.

# It includes multiple CSV files with:

# final_list: Talk IDs, titles, speaker names, and URLs

# details: Descriptions, views, duration

# tags: Talk themes

# related_videos, images, and more

# Licensed under MIT, allowing flexible use.