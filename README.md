# AI Image Retrieval System

Search images using natural language queries powered by AI vision and vector embeddings.

## Overview

This system transforms images into searchable data using:
- **Ministral-3:3b** (Vision AI) - Generates text descriptions from images
- **EmbeddingGemma** (Embedding Model) - Converts text to 768-dimensional vectors
- **Cosine Similarity** - Finds semantically similar images

**Example:** Search for "bear in forest" → System finds all images with bears, wildlife, or natural scenes.

## Quick Start

### 1. Install Requirements

**Ollama Models:**
```bash
ollama pull ministral-3:3b        # Vision model for image description
ollama pull embeddinggemma:latest # Embedding model for vectors
```

**Python Dependencies:**
```bash
pip install pandas numpy ollama gradio
```

### 2. Setup Your Images

Place your JPG images in the `frames/` folder (one level up from this directory):
```
Image_Retrieval_System/
├── frames/                      # Put your images here
│   ├── bear.jpg
│   ├── bird.jpg
│   └── ...
└── Image_Retrieval_System/      # Project files
    ├── app.py                   # Web interface (Gradio)
    ├── final_search.py          # Command-line search
    ├── pipeline.py              # Image indexing script
    ├── Image_Retrieval_System.ipynb  # Jupyter notebook
    ├── model.pkl                # Vector database (created after indexing)
    └── README.md                # This file
```

### 3. Index Your Images

```bash
python pipeline.py
```

**What happens:**
1. Scans `frames/` folder for all JPG images
2. Each image is analyzed by Vision AI → generates description
3. Description converted to 768-dimensional vector by Embedding model
4. All data saved to `model.pkl` database
5. Already-indexed images are automatically skipped

**Console output:**
```
🔎 Found 10 images. Checking for new files...
Processing: /path/to/bear.jpg
Success! Total images in model: 10
```

### 4. Search Your Images

**Option A: Command Line Interface**
```bash
python final_search.py
```

Features:
- Interactive terminal search
- Shows vector dimensions and sample values
- Returns top 3 matches with confidence scores
- Fast and lightweight

**Example session:**
```
DATABASE LOADED: 10 images indexed.
VECTOR DIMENSIONS: 768 (Gemma Standard)

Enter search term: wildlife in nature

TOP 3 MATCHES FOR: 'wildlife in nature'
1. FILE: bear.jpg
   CONFIDENCE: 87.45%
   DESCRIPTION: Grizzly bear, Rocky landscape, Natural forest
```

**Option B: Web Interface (Gradio)**
```bash
python app.py
```

Features:
- Modern web UI (auto-opens in browser)
- **Search Engine Tab** - Visual search with image preview
- **Database Explorer Tab** - View all indexed images
- Real-time confidence scoring
- No need to type file paths

## Project Structure

| File | Purpose | Details |
|------|---------|---------|
| `pipeline.py` | Image indexing pipeline | Scans images, generates descriptions, creates embeddings, saves to model.pkl |
| `final_search.py` | Command-line search | Interactive terminal search with built-in vector inspector |
| `app.py` | Web interface | Gradio-based GUI with image preview and database explorer |
| `model.pkl` | Vector database | Pandas DataFrame: [filename, description, 768-dim embedding] |
| `Image_Retrieval_System.ipynb` | Jupyter notebook | Step-by-step walkthrough of the entire pipeline |

## How It Works

### Pipeline Flow
```
1. Image File (.jpg)
   ↓
2. Vision AI (Ministral-3:3b)
   → Analyzes image
   → Output: "Grizzly bear in natural forest setting"
   ↓
3. Embedding Model (EmbeddingGemma)
   → Converts text to numbers
   → Output: [0.05, -0.02, 0.03, ..., 0.01] (768 values)
   ↓
4. Database (model.pkl)
   → Stores: filename + description + vector
```

### Search Process
```
User Query: "wildlife animal"
   ↓
1. Convert query to vector using EmbeddingGemma
   → [0.04, -0.03, 0.02, ..., 0.02]
   ↓
2. Compare with all image vectors using Cosine Similarity
   → Scores: 0.87, 0.65, 0.43, ... (1=perfect match, 0=no match)
   ↓
3. Return top matches ranked by score
```

### Key Technologies

**Vector Embeddings (768 dimensions):**
- Each image becomes 768 numbers representing its semantic meaning
- Similar concepts = similar vectors
- Example:
  - "bear in forest" → `[0.05, -0.02, 0.03, ...]`
  - "wildlife nature" → `[0.04, -0.03, 0.02, ...]` ← Similar!
  - "city building" → `[-0.08, 0.09, -0.01, ...]` ← Different

**Cosine Similarity:**
```python
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
- Measures angle between vectors (not distance)
- Returns score from 0 to 1
- 1.0 = identical meaning
- 0.5 = somewhat related
- 0.0 = completely unrelated

## Updating Your Database

### Add New Images
```bash
# 1. Add new JPG files to frames/ folder
# 2. Re-run pipeline (only processes new images)
python pipeline.py
```
Output: `"Database is already up to date."` or `"Success! Total images in model: 15"`

### Rebuild From Scratch
```bash
rm model.pkl
python pipeline.py
```

### Check What's Indexed
Run `final_search.py` - it displays vector info on startup, or check the Database Explorer tab in `app.py`

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| **"model.pkl not found"** | Database not created yet | Run `python pipeline.py` first |
| **"No images found"** | Empty frames folder | Add JPG images to `frames/` folder |
| **"Ollama connection error"** | Ollama not running | Start Ollama: `ollama serve` |
| **"No module named 'pandas'"** | Missing dependencies | Run `pip install pandas numpy ollama gradio` |
| **Poor search results** | Bad embeddings or descriptions | Check vector info in `final_search.py` output |
| **Search returns nothing** | Query too specific | Try broader terms like "animal" instead of "brown bear" |

## Tips for Best Results

**Search Query Tips:**
- Use descriptive terms: `"wildlife in nature"` not `"pic1"`
- Try broader categories: `"animal"` before `"grizzly bear"`
- Combine concepts: `"sunset beach ocean"`

**Image Indexing Tips:**
- Use clear, high-quality images
- Consistent naming helps organization
- Re-index if you update many images

## Technical Specifications

- **Vector Dimensions:** 768 (EmbeddingGemma standard)
- **Image Format:** JPG only
- **Indexing Speed:** ~2-5 seconds per image (depends on hardware)
- **Search Speed:** ~0.5-1 second per query
- **Database Format:** Pickle (pandas DataFrame)
- **Storage:** ~6KB per indexed image (description + embedding)

## Example Use Cases

1. **Photo Library:** Search `"family gathering"` across thousands of photos
2. **Wildlife Research:** Find `"endangered species"` in camera trap images
3. **E-commerce:** Search `"red shoes"` in product catalogs
4. **Medical Imaging:** Locate `"specific condition"` in X-rays/scans

---

**Built with using Ollama • Ministral-3:3b • EmbeddingGemma • Gradio**
