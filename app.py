import gradio as gr
import pandas as pd
import numpy as np
import ollama
import os

# --- 1. CONFIGURATION & DATA LOADING ---
MODEL_PATH = "model.pkl"
# Get absolute path for the frames directory for Gradio security
FRAMES_DIR = os.path.abspath("../frames")

if not os.path.isfile(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found. Please run pipeline.py first.")
    exit()

# Load the database
df = pd.read_pickle(MODEL_PATH)

def search_images(query):
    if not query.strip():
        return None, "0%", "Please enter a search term."

    try:
        # Generate embedding for the search query
        res = ollama.embed(model='embeddinggemma:latest', input=query)
        q_vec = np.array(res['embeddings'][0])

        # Cosine Similarity calculation
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Calculate scores
        df["score"] = df["embedding"].apply(lambda x: cos_sim(np.array(x), q_vec))
        
        # Get the top matching image
        top_match = df.sort_values("score", ascending=False).iloc[0]
        
        return top_match['filename'], f"{top_match['score']:.2%}", top_match['description']

    except Exception as e:
        return None, "Error", f"Search failed: {str(e)}"

def get_db_summary():
    """Helper to show the contents of the pickle file in the UI"""
    inspect_df = df.copy()
    inspect_df['filename'] = inspect_df['filename'].apply(lambda x: os.path.basename(x))
    inspect_df['vector_peek'] = inspect_df['embedding'].apply(lambda x: str(x[:5]) + "...")
    return inspect_df[['filename', 'description', 'vector_peek']]

# --- 2. GRADIO INTERFACE DESIGN ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎬 AI Video Frame Search Engine")
    
    with gr.Tabs():
        # TAB 1: THE SEARCH INTERFACE
        with gr.TabItem("🔍 Search Engine"):
            with gr.Row():
                # Left Column: Inputs
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label="Search Query", 
                        placeholder="Describe a scene...",
                        lines=2
                    )
                    search_button = gr.Button("Find Matching Frame", variant="primary")
                    conf_output = gr.Label(label="AI Confidence Score")
                
                # Right Column: Visual Results
                with gr.Column(scale=2):
                    img_output = gr.Image(label="Best Visual Match", type="filepath")
                    # Increased box size: lines=10 makes it tall and readable
                    desc_output = gr.Textbox(
                        label="AI Frame Analysis", 
                        lines=10, 
                        max_lines=15
            )
                    

            # Link the search logic
            search_button.click(
                fn=search_images, 
                inputs=query_input, 
                outputs=[img_output, conf_output, desc_output]
            )

        # TAB 2: DATABASE INSPECTOR (Pickle Viewer)
        with gr.TabItem("📦 Database Explorer"):
            gr.Markdown(f"### Internal Metadata (Total Frames: {len(df)})")
            db_table = gr.Dataframe(value=get_db_summary(), interactive=False)

# --- 3. LAUNCH THE APPLICATION ---
if __name__ == "__main__":
    print(f"🚀 Launching Search Engine...")
    
    # Final Settings: Theme, Security (allowed_paths), and Auto-Launch (inbrowser)
    demo.launch(
        theme=gr.themes.Soft(),
        allowed_paths=[FRAMES_DIR],
        inbrowser=True  # Automatically opens Safari/Chrome for you
    )