# 🎬 CineMatch | Research Portfolio & Recommendation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](#)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CineMatch is a high-performance, content-based movie recommender system designed as a research portfolio. It combines advanced natural language processing (NLP) with a premium user interface to demonstrate the power of vector similarity in information retrieval.

---

## 🔬 Core Features

### 1. Vector Similarity Engine
*   **Methodology**: Uses **Cosine Similarity** on high-dimensional movie metadata vectors.
*   **Tag Fusion**: Combines plot overviews, genres, keywords, cast, and directors into unique textual signatures.
*   **3D Manifold Visualization**: Projects the 5000-D feature space into a interactive 3D manifold using **Truncated SVD** for latent cluster analysis.

### 2. Explainable AI (XAI)
*   **Transparency**: Every recommendation includes a mathematical breakdown of why it was chosen.
*   **Feature Attribution**: Visualizes shared metadata features between the query and recommended movies.

### 3. Experimental Evaluation
*   **Ablation Study**: Includes a module to benchmark different vectorization strategies (**Bag-of-Words**, **TF-IDF**, and **SBERT**).
*   **Precision@10**: Evaluates accuracy using Genre Overlap as a scientific proxy for relevance.

### 4. Enterprise-Grade Reliability
*   **API Load Balancing**: Implements random round-robin rotation between multiple TMDB API keys to maximize image loading throughput.
*   **Resilient Fallbacks**: Multi-stage poster fetching strategy (Direct ID -> Title Search -> High-end Placeholder).

---

## 🛠️ Project Structure

```text
├── Movie.py                # Main Streamlit Application UI & Logic
├── app_utils.py            # API Hooks & Utility Functions (Load Balanced)
├── config.py               # Centralized Configuration & API Management
├── preprocess.py           # Research Pipeline (Data Fusion -> SVD Projection)
├── evaluation.py           # Experimental Benchmarking Module
├── explainability.py       # XAI Feature Attribution Module
├── style.css               # Premium "Neon Tech" UI Design System
└── requirements.txt        # System Dependencies
```

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/yourusername/CineMatch.git
cd CineMatch
pip install -r requirements.txt
```

### 2. Setup Data
Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place the CSVs in the root directory. Then run the research pipeline:
```bash
python preprocess.py
```

### 3. Local Deployment
```bash
streamlit run Movie.py
```

---

## ⚙️ Configuration

CineMatch is designed for easy deployment. For Streamlit Cloud, add your API keys to the Secrets management:

```toml
[api]
TMDB_API_KEY = "your_primary_key"
TMDB_API_KEY_BACKUP = "your_secondary_key"
```

---

## 👨‍💻 Author
**K RAMA KRISHNA NARASIMHA CHOWDARY**  
*Research Status: ONLINE*

---
© 2026 CineMatch Research Project. Released under the MIT License.
