# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import ollama
import numpy as np
import os
import glob

# --- CONFIGURATION ---
IMAGE_FOLDER = "/Users/sayemaraf/Desktop/BIG Data projects/Image_Retrieval_System/frames"  
MODEL_FILE = "model.pkl"

def run_pipeline():
    # 1. Automated Discovery: Find all JPG files
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    if not image_paths:
        print(f"Error: No images found in '{IMAGE_FOLDER}'.")
        return

    print(f"🔎 Found {len(image_paths)} images. Checking for new files...")

    # 2. Database Handling: Load existing data or start fresh
    if os.path.exists(MODEL_FILE):
        df = pd.read_pickle(MODEL_FILE)
    else:
        df = pd.DataFrame(columns=['filename', 'description', 'embedding'])

    new_entries = []

    # 3. Batch Processing
    for path in image_paths:
        # Skip if already in the database to save time
        if path in df['filename'].values:
            continue
            
        print(f"Processing: {path}")
        try:
            # Task A: Describe image using Ministral-3:3b (Vision)
            res_vision = ollama.chat(model='ministral-3:3b', messages=[
                {'role': 'user', 'content': 'Describe this image in 5 words.', 'images': [path]}
            ])
            desc = res_vision['message']['content'].strip()
            
            # Task B: Get vector embedding using EmbeddingGemma
            res_embed = ollama.embed(model='embeddinggemma:latest', input=desc)
            vector = np.array(res_embed['embeddings'][0])
            
            new_entries.append({'filename': path, 'description': desc, 'embedding': vector})
        except Exception as e:
            print(f"Error on {path}: {e}")

    # 4. Save and Finish
    if new_entries:
        df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
        df.to_pickle(MODEL_FILE)
        print(f"Success! Total images in model: {len(df)}")
    else:
        print("Database is already up to date.")

if __name__ == "__main__":
    run_pipeline()
