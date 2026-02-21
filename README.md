# CineMatch | Research Portfolio & Recommendation Engine

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?logo=github)](#)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CineMatch is a high-performance, content-based movie recommender system designed as a research portfolio. It combines advanced natural language processing (NLP) with a premium user interface to demonstrate the power of vector similarity in information retrieval.

**Live Demo**: [https://kram2006.github.io/cinematch/](https://kram2006.github.io/cinematch/)

---

## Core Features

### 1. Vector Similarity Engine
*   **Methodology**: Uses **Cosine Similarity** on high-dimensional movie metadata vectors.
*   **Tag Fusion**: Combines plot overviews, genres, keywords, cast, and directors into unique textual signatures.
*   **3D Manifold Visualization**: Projects the 5000-D feature space into an interactive 3D manifold using **Truncated SVD** for latent cluster analysis.

### 2. Explainable AI (XAI)
*   **Transparency**: Every recommendation includes a mathematical breakdown of why it was chosen.
*   **Feature Attribution**: Visualizes shared metadata features between the query and recommended movies.

### 3. Experimental Evaluation
*   **Ablation Study**: Benchmarks different vectorization strategies (**Bag-of-Words**, **TF-IDF**, and **SBERT**).
*   **Precision@10**: Evaluates accuracy using Genre Overlap as a scientific proxy for relevance.

### 4. Enterprise-Grade Reliability
*   **API Load Balancing**: Implements random round-robin rotation between multiple TMDB API keys to maximize image loading throughput.
*   **Resilient Fallbacks**: Multi-stage poster fetching strategy (Direct ID -> Title Search -> High-end Placeholder).

---

## Project Structure

```text
├── index.html              # Main Static Site (GitHub Pages)
├── app.js                  # Client-Side Application Logic (Core Engine)
├── style.css               # Premium "Neon Tech" UI Design System
├── data/
│   ├── movies.json         # Exported Movie List (4,806 titles)
│   ├── recommendations.json # Pre-computed Top-10 Recommendations
│   ├── manifold.json       # 3D SVD Coordinates
│   └── evaluation.json     # Pre-computed Benchmark Metrics
├── app_utils.py            # API Hooks & Metadata Utilities
├── config.py               # Configuration & API Key Rotation
├── preprocess.py           # Research Pipeline (Vectorization -> SVD)
├── evaluation.py           # Experimental Benchmarking Module
├── explainability.py       # XAI Feature Attribution Module
├── export_data.py          # Pickle -> JSON Data Export Tool
├── similarity.pkl          # Pre-computed Similarity Matrix (Large)
├── movie_list.pkl          # Processed Movie Metadata Pickle
├── cv.pkl                  # Fitted Vectorizer Model
└── requirements.txt        # Python Dependencies (for pipeline execution)
```

---

## Deployment

### GitHub Pages (Live Site)
The site is deployed automatically via GitHub Pages from the `main` branch. Simply push to `main` and the site updates.

### Regenerating Data (Advanced)
If you need to regenerate the recommendation data from scratch:

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place CSVs in root.
2. Run the research pipeline:
   ```bash
   pip install -r requirements.txt
   python preprocess.py
   python export_data.py
   ```

---

## Author
**K RAMA KRISHNA NARASIMHA CHOWDARY**  
*Research Status: ONLINE*

---
© 2026 CineMatch Research Project. Released under the MIT License.
