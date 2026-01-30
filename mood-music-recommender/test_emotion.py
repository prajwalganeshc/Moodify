from transformers import pipeline
from recommender import recommend_songs

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"], round(result["score"], 3)

if __name__ == "__main__":
    user_input = input("How are you feeling today? → ")
    label, confidence = detect_emotion(user_input)
    print(f"Detected Emotion: {label} ({confidence * 100}%)")

    print("\n🎵 Recommended Songs:")
    recommendations = recommend_songs(label.lower())
    print(recommendations.to_string(index=False))