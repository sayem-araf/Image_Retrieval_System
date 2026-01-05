# %%
import os
import numpy as np
import pandas as pd
import ollama

MODEL_PATH = "model.pkl"
# TIP: If your frames are in the same folder as the script, change to "./frames"
FRAMES_DIR = os.path.abspath("../frames")

if not os.path.isfile(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    exit()

df = pd.read_pickle(MODEL_PATH)

# --- START VECTOR INSPECTOR SECTION ---
print("="*50)
print(f"✅ DATABASE LOADED: {len(df)} images indexed.")

# Check the first row to inspect the math
first_vector = np.array(df.iloc[0]['embedding'])
print(f"📍 VECTOR DIMENSIONS: {first_vector.shape[0]} (Gemma Standard)")
print(f"📍 SAMPLE EMBEDDING (First 3 values): {first_vector[:3]}")
print("="*50)
# --- END VECTOR INSPECTOR SECTION ---

query = input("\nEnter search term: ")

if not query.strip():
    exit()

print(f"🛰️ AI is searching for: '{query}'...")

try:
    # Use the newer 'embed' method which matches your environment
    res = ollama.embed(model='embeddinggemma:latest', input=query)
    q_vec = np.array(res['embeddings'][0])

    def cos_sim(a, b):
        # Calculate cosine similarity between the query and stored vectors
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    df["score"] = df["embedding"].apply(lambda x: cos_sim(np.array(x), q_vec))
    results = df.sort_values("score", ascending=False).head(3)

    print("\n" + "="*50)
    print(f"TOP 3 MATCHES FOR: '{query}'")
    print("="*50)

    for i, (idx, row) in enumerate(results.iterrows(), 1):
        print(f"{i}. FILE: {row['filename']}")
        print(f"   CONFIDENCE: {row['score']:.2%}")
        print(f"   DESCRIPTION: {row['description']}")
        print("-" * 50)

except Exception as e:
    print(f" An error occurred: {e}")