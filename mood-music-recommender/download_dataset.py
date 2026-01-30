from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("maharshipandya/spotify-tracks-dataset")

# Convert to pandas DataFrame
df = dataset["train"].to_pandas()

# Save as CSV
df.to_csv("spotify_tracks.csv", index=False)

print("✅ Dataset downloaded and saved as spotify_tracks.csv")
