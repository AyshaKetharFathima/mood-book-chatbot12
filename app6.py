import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER sentiment lexicon (only first run)
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# --- Sample Dataset ---
books = [
    {"title": "The Happiness Project", "description": "A guide to finding joy and happiness in daily life", "tags": "happiness positive self-help"},
    {"title": "The Fault in Our Stars", "description": "A touching romance about illness and love", "tags": "romance tragedy sad"},
    {"title": "Calm Your Mind", "description": "Relaxation and meditation for stress relief", "tags": "calm meditation peace"},
    {"title": "Deep Work", "description": "Techniques for focus and productivity", "tags": "focus work productivity"},
    {"title": "The Party Crasher", "description": "A fun romantic comedy full of laughter", "tags": "fun comedy romance"},
    {"title": "Thinking Fast and Slow", "description": "Psychology and insights into decision making", "tags": "psychology thinking mind"}
]

df = pd.DataFrame(books)
df["text"] = df["title"] + " " + df["description"] + " " + df["tags"]

# TF-IDF setup
vectorizer = TfidfVectorizer(stop_words="english")
book_vectors = vectorizer.fit_transform(df["text"])

# Mood-to-tags mapping
mood_tag_map = {
    "happy": "happiness positive fun comedy",
    "sad": "sad tragedy loss romance",
    "relaxed": "calm meditation peace self-help",
    "focus": "focus work productivity",
    "thoughtful": "psychology thinking decision-making"
}

def mood_to_vector(mood_text):
    # Step 1: Sentiment Analysis
    scores = sia.polarity_scores(mood_text)
    if scores["compound"] >= 0.5:
        mood = "happy"
    elif scores["compound"] <= -0.5:
        mood = "sad"
    else:
        # keyword matching for neutral cases
        text_lower = mood_text.lower()
        if "focus" in text_lower or "study" in text_lower:
            mood = "focus"
        elif "calm" in text_lower or "relax" in text_lower:
            mood = "relaxed"
        elif "think" in text_lower or "deep" in text_lower:
            mood = "thoughtful"
        else:
            mood = "happy"  # default fallback

    pseudo_doc = mood_tag_map[mood]
    return vectorizer.transform([pseudo_doc]), mood

def recommend_books(user_text, top_k=3):
    mood_vec, detected_mood = mood_to_vector(user_text)
    sims = cosine_similarity(mood_vec, book_vectors).flatten()
    df["score"] = sims
    recs = df.sort_values("score", ascending=False).head(top_k)
    return recs[["title", "description"]], detected_mood

# --- Streamlit Chatbot UI ---
st.set_page_config(page_title="📚 Mood-Based Book Chatbot", page_icon="📖")

st.title("📚 Mood-Based Book Chatbot")
st.write("Chat with me about how you feel, and I’ll recommend books!")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for role, msg in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(msg)

# User input
if user_input := st.chat_input("Type how you feel..."):
    # Add user message
    st.session_state["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Generate recommendations
    recs, mood = recommend_books(user_input)

    bot_reply = f"I sense you are feeling **{mood}**. Here are some book suggestions:\n"
    for _, row in recs.iterrows():
        bot_reply += f"📖 **{row['title']}** – {row['description']}\n\n"

    # Add bot message
    st.session_state["messages"].append(("assistant", bot_reply))
    with st.chat_message("assistant"):
        st.write(bot_reply)
