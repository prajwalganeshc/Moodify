import pandas as pd

songs_df = pd.read_csv("spotify_tracks.csv")

valence_map = {
    "joy" : (0.6, 0.1),
    "love" : (0.6, 0.1),
    "surprise" : (0.5, 0.9),
    "sadness" : (0.0, 0.3), 
    "anger" : (0.3,  0.5),
    "fear" : (0.2, 0.5),
    "nuetral" : (0.4, 0.6)
}

def recommend_songs(emotion, top_n = 5):
    if emotion not in valence_map:
        print(f"⚠️ Unsupported emotion: {emotion}")
        return []
    
    val_min, val_max = valence_map[emotion]

    mood_songs = songs_df[(songs_df["valence"] >= val_min)&(songs_df["valence"] <= val_max)]

    if mood_songs.empty:
        print("😢 No songs found for this mood. Showing random ones instead.")
        mood_songs = songs_df.sample(n=top_n)

    
    return mood_songs[["track_name", "artists", "valence"]].head(top_n)
