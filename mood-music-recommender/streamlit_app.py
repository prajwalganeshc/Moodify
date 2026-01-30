import streamlit as st
from transformers import pipeline
from recommender import recommend_songs
import pandas as pd

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model = "j-hartmann/emotion-english-distilroberta-base",
        return_all_scores = False
    )

emotion_classifier = load_emotion_model()

st.set_page_config(
    page_title="Moodify 🎧",
    page_icon="🎵",
    layout="centered"
)


st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎧 Moodify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Discover music that matches your emotions</h4>", unsafe_allow_html=True)

st.write("Type how you're feeling and get songs that match your emotion!")

st.markdown("#### 💭 How are you feeling today?")
user_input = st.text_area("", height=100, placeholder="Type something like: I'm feeling energetic and happy...")


if st.button("Get Songs 🎵") and user_input:
    with st.spinner("Detecting emotion..."):
        result = emotion_classifier(user_input)[0]
        label = result["label"]
        confidence = round(result["score"],3)

        st.success(f"Detected Emotion:{label}({confidence * 100}%)")

        st.subheader("🎵 Recommended Songs:")
        recs = recommend_songs(label.lower(),top_n=5)


        if isinstance(recs, pd.DataFrame) and not recs.empty:

            for i, row in recs.iterrows():
                st.markdown(f"""
        <div style="padding: 10px; border-radius: 10px; background-color: #f4f4f4; margin-bottom: 10px;">
            <strong>{row['track_name']}</strong><br>
            <span style='color: gray;'>by {row['artists']}</span><br>
            <span style='color: #4CAF50;'>Valence: {round(row['valence'], 2)}</span>
        </div>
        """, unsafe_allow_html=True)


        else:
            st.warning("No songs found for this emotion. My bad gng!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with 💚 using Streamlit and Hugging Face</p>", unsafe_allow_html=True)

# Virtual Enviroment Activation : .\.venv\Scripts\Activate.ps1
# Run : streamlit run streamlit_app.py