import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Expanded dataset
# ---------------------------
items = [
    {"item_id": 1, "title": "Sunny Morning", "description": "upbeat pop cheerful guitar", "tags": "pop upbeat guitar"},
    {"item_id": 2, "title": "Late Night City", "description": "chill electronic mellow ambient", "tags": "electronic chill ambient"},
    {"item_id": 3, "title": "Heartbreak Ballad", "description": "slow sad piano emotional", "tags": "ballad sad piano"},
    {"item_id": 4, "title": "Focus Flow", "description": "instrumental minimal concentration", "tags": "instrumental focus minimal"},
    {"item_id": 5, "title": "Party Anthem", "description": "energetic dance club vibes", "tags": "dance energetic club"},
    {"item_id": 6, "title": "Cozy Fireplace", "description": "warm acoustic calm vocals", "tags": "acoustic calm cozy"},
    {"item_id": 7, "title": "Rainy Reflections", "description": "emotional violin soft piano", "tags": "sad emotional violin"},
    {"item_id": 8, "title": "Zen Garden", "description": "relaxing calm meditation vibes", "tags": "relax calm meditation"},
    {"item_id": 9, "title": "Study Beats", "description": "instrumental lo-fi concentration", "tags": "focus lofi instrumental"},
    {"item_id": 10, "title": "Festival Night", "description": "party loud edm dance", "tags": "party edm dance"},
]

df = pd.DataFrame(items)
df["text"] = df["title"] + " " + df["description"] + " " + df["tags"]

# ---------------------------
# 2. TF-IDF setup
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")
item_vectors = vectorizer.fit_transform(df["text"])

# ---------------------------
# 3. Mood-to-tag mapping
# ---------------------------
mood_tag_map = {
    "happy": {"upbeat": 1.0, "energetic": 0.9, "pop": 0.8, "dance": 0.7},
    "sad": {"sad": 1.0, "ballad": 0.9, "piano": 0.8, "emotional": 0.7},
    "relaxed": {"chill": 1.0, "ambient": 0.9, "calm": 0.8, "acoustic": 0.7, "meditation": 0.9},
    "focus": {"focus": 1.0, "instrumental": 0.9, "minimal": 0.8, "lofi": 0.8},
    "party": {"club": 1.0, "energetic": 0.95, "dance": 0.9, "edm": 0.9},
}

def mood_to_vector(mood):
    """Convert mood into a pseudo-document and transform into TF-IDF vector"""
    if mood not in mood_tag_map:
        raise ValueError(f"Mood '{mood}' not supported.")
    tags = mood_tag_map[mood]
    # Weighted pseudo-document
    pseudo_doc = " ".join([tag for tag, w in tags.items() for _ in range(int(w * 10))])
    return vectorizer.transform([pseudo_doc])

# ---------------------------
# 4. Recommendation function
# ---------------------------
def recommend(mood, top_k=3):
    mood_vec = mood_to_vector(mood)
    sims = cosine_similarity(mood_vec, item_vectors).flatten()
    df["score"] = sims
    results = df.sort_values(by="score", ascending=False).drop_duplicates(subset=["title"]).head(top_k)
    return results[["title", "score"]]

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Mood-Based Recommender", page_icon="📚")

st.title("🎵 Mood-Based Recommender")
st.write("Tell me your mood and I’ll recommend music/books!")

# Dropdown for moods
mood = st.selectbox("Choose your mood:", ["happy", "sad", "relaxed", "focus", "party"])

if st.button("Get Recommendations"):
    try:
        recs = recommend(mood)
        st.subheader(f"Recommendations for mood: **{mood.capitalize()}**")
        for i, row in recs.iterrows():
            st.write(f"📖 **{row['title']}** (Score: {row['score']:.2f})")
    except Exception as e:
        st.error(str(e))
