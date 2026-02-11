import requests
import wikipediaapi
import streamlit as st
import pandas as pd
import numpy as np
import logging
import os

from config import WIKI_USER_AGENT, setup_logging, TMDB_API_KEYS

# Setup logging
logger = setup_logging()

# Correct Caching Decorator
@st.cache_data(ttl=3600)
def get_wiki_data(movie_title):
    """
    Fetch Wikipedia summary for a movie.
    
    Args:
        movie_title: The movie title to search for
        
    Returns:
        dict: Contains 'summary' and 'url' keys
    """
    try:
        # Fixed: Correct Wikipedia API initialization
        wiki = wikipediaapi.Wikipedia(
            user_agent=WIKI_USER_AGENT,
            language='en'
        )
        
        page = wiki.page(movie_title)
        if page.exists():
            summary = page.summary[:800]
            if len(page.summary) > 800:
                summary += "..."
            return {
                'summary': summary,
                'url': page.fullurl
            }
        
        # Try appending (film) if direct lookup fails
        page_film = wiki.page(f"{movie_title} (film)")
        if page_film.exists():
            summary = page_film.summary[:800]
            if len(page_film.summary) > 800:
                summary += "..."
            return {
                'summary': summary,
                'url': page_film.fullurl
            }
        
        logger.info(f"No Wikipedia entry found for: {movie_title}")
        return {'summary': "No Wikipedia entry found.", 'url': '#'}
        
    except Exception as e:
        logger.error(f"Wikipedia lookup error for '{movie_title}': {str(e)}")
        return {'summary': "Wiki lookup unavailable.", 'url': '#'}

def get_wiki_summary(movie_title):
    """
    Get Wikipedia summary text for a movie.
    
    Args:
        movie_title: The movie title
        
    Returns:
        str: Wikipedia summary text
    """
    data = get_wiki_data(movie_title)
    return data['summary']

def fetch_movie_details(movie_id):
    """
    Fetch detailed movie information from TMDB API with multi-key rotation and balanced workload.
    
    Args:
        movie_id: TMDB movie ID
        
    Returns:
        dict: Movie details including rating, release_date, runtime, overview, trailer
        None: If request fails
    """
    from config import TMDB_API_KEYS
    import random
    shuffled_keys = TMDB_API_KEYS.copy()
    random.shuffle(shuffled_keys)
    
    for key in shuffled_keys:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}&append_to_response=videos"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                
                details = {
                    'rating': data.get('vote_average', 'N/A'),
                    'release_date': data.get('release_date', 'N/A'),
                    'runtime': f"{data.get('runtime', 0)} min",
                    'overview': data.get('overview', ''),
                    'trailer': None
                }
                
                videos = data.get('videos', {}).get('results', [])
                for video in videos:
                    if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                        details['trailer'] = f"https://www.youtube.com/watch?v={video['key']}"
                        break
                
                return details
            
            # Switch key on rate limit
            if res.status_code == 429:
                continue
                
        except:
            continue
            
    return None

def analyze_vibe(text):
    """Simple rule-based 'vibe' analysis for portfolio show-off."""
    positive_words = ['exciting', 'love', 'hero', 'fun', 'happy', 'adventure', 'great', 'brilliant', 'wonderful', 'joy']
    dark_words = ['death', 'murder', 'scary', 'blood', 'terror', 'dark', 'evil', 'ghost', 'war', 'tragedy']
    
    words = text.lower().split()
    pos_score = sum(1 for w in words if w in positive_words)
    dark_score = sum(1 for w in words if w in dark_words)
    
    if dark_score > pos_score: return "DARK / GRITTY"
    if pos_score > dark_score: return "UPLIFTING / HEROIC"
    return "BALANCED / MYSTERIOUS"

def get_pipeline_data():
    return [
        {
            "step": "01 // DATA HARVESTING",
            "process": "Ingesting TMDB 5,000 dataset (Movies + Credits).",
            "transformation": "Shape: (4803, 20) + (4803, 4) -> Merged on Title",
            "technical": "Handling JSON-serialized strings (Genres, Keywords, Cast) via ast.literal_eval.",
            "color": "#ff3e3e"
        },
        {
            "step": "02 // FEATURE ENGINEERING",
            "process": "Extraction of high-signal metadata.",
            "transformation": "Cast (Top 3), Director, Keywords, Genres.",
            "technical": "Removing whitespace to create unique semantic tokens (e.g., 'Science Fiction' -> 'sciencefiction').",
            "color": "#ffcc00"
        },
        {
            "step": "03 // FEATURE FUSION",
            "process": "Concatenation of all metadata into a weighted 'Tag Cloud'.",
            "transformation": "Overview + Genres + Keywords + Cast + Director -> Unified String.",
            "technical": "NLP normalization (Lowercasing, Stop-word removal).",
            "color": "#00ff41"
        },
        {
            "step": "04 // VECTORIZATION",
            "process": "Transforming text into a high-dimensional sparse matrix.",
            "transformation": "Unified String -> 5,000-D Bag-of-Words Vector.",
            "technical": "Sklearn CountVectorizer(max_features=5000). Toarray() conversion for similarity scanning.",
            "color": "#38bdf8"
        },
        {
            "step": "05 // MATHEMATICAL CORE",
            "process": "Computing the angular similarity between movie vectors.",
            "transformation": "5000-D Space -> (4806 x 4806) Symmetric Matrix.",
            "technical": "Cosine Similarity: Realized as float32 for 50% memory efficiency (90MB footprint).",
            "color": "#818cf8"
        },
        {
            "step": "06 // MANIFOLD PROJECTION",
            "process": "Visualizing the latent similarity manifold.",
            "transformation": "4806-D Similarity Rows -> 3D Latent Coordinates (X, Y, Z).",
            "technical": "Truncated SVD (Singular Value Decomposition) preserves global distance relationships.",
            "color": "#c084fc"
        },
        {
            "step": "07 // REAL-TIME INFERENCE",
            "process": "Generating recommendations via Nearest Neighbor search.",
            "transformation": "Input Title -> Top 6 Correlated Movies.",
            "technical": "Multi-threaded TMDB API calls + Wikipedia Context + Feature Attribution (XAI).",
            "color": "#f472b6"
        }
    ]

def get_methodology_content():
    return {
        'concept': """### CONTENT-BASED FILTERING ARCHITECTURE
The core of CineMatch follows a **Content-Based Filtering** approach. This system analyzes the intrinsic properties of movies (metadata) to build a user-agnostic recommendation engine.""",
        'deep_research': """### DEEP RESEARCH: DIMENSIONALITY & LATENT SPACES
In modern Recommender Systems, we deal with **High-Dimensional Sparse Matrices**. 

**1. The Curse of Dimensionality**: With 5,000 unique tags, each movie is a point in a 5,000-D space. Calculating similarity here is computationally expensive.

**2. Latent Factors**: While this version uses a Bag-of-Words (BoW) approach, industry leaders like Netflix use **Latent Factor Models**. This involves decomposing the matrix (SVD/Matrix Factorization) to find hidden "themes" that explain user preferences without needing explicit tags.

**3. NLP Evolution (Research Notes)**:
- **Stage 1 (BoW)**: Counting words (Current Implementation).
- **Stage 2 (TF-IDF)**: Weighting rarity.
- **Stage 3 (Word Embeddings)**: Using BERT or Word2Vec to understand that "Space" and "Cosmos" are semantically identical, even if the words are different.""",
        'comparison': """### TECHNICAL RATIONALE: BAG OF WORDS vs. TF-IDF
While this implementation utilizes **CountVectorizer** for its deterministic behavior, **TF-IDF** (Term Frequency-Inverse Document Frequency) is a robust research alternative.

- **Decision**: For this dataset, CountVectorizer was selected to maintain the high signal weight of genre tags which appear frequently across the corpus but are critical for matching.""",
        'steps': [
            {
                'title': 'DATA HARMONIZATION',
                'desc': 'Merging TMDB 5000 datasets to create a centralized feature set.',
                'code': "movies = movies.merge(credits, on='title')"
            },
            {
                'title': 'VECTORIZATION STEP',
                'desc': 'Applying NLP tokenization to convert text into a 5000-dimensional vector space.',
                'code': "cv = CountVectorizer(max_features=5000, stop_words='english')"
            }
        ],
        'math': """### SEMANTIC PROXIMITY: WHY COSINE SIMILARITY?
$$ \\text{Cosine Similarity} = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|} $$
Cosine similarity is preferred over Euclidean distance for text data because it remains invariant to **document length**. In a movie tag cloud, the *content* (direction) is more important than the *length* of the summary (magnitude)."""
    }
