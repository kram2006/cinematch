import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import ast
import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CineMatch.preprocess')

def load_tmdb_data():
    """
    Load and merge TMDB 5000 datasets.
    
    Returns:
        DataFrame: Merged movies and credits data
        None: If files not found
    """
    required_files = ['tmdb_5000_movies.csv', 'tmdb_5000_credits.csv']
    
    # Check if files exist
    for filename in required_files:
        if not os.path.exists(filename):
            logger.error(f"Required file not found: {filename}")
            logger.info("Please download TMDB 5000 dataset from:")
            logger.info("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
            return None
    
    try:
        logger.info("Loading TMDB datasets...")
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        logger.info(f"Loaded {len(movies)} movies and {len(credits)} credits")
        
        # Merge on title
        movies = movies.merge(credits, on='title')
        logger.info(f"Merged dataset shape: {movies.shape}")
        
        return movies
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def extract_features(obj):
    """
    Extract names from JSON-like string.
    
    Args:
        obj: JSON string or list
        
    Returns:
        list: Extracted names
    """
    try:
        if isinstance(obj, str):
            parsed = ast.literal_eval(obj)
        else:
            parsed = obj
            
        if isinstance(parsed, list):
            return [item['name'] for item in parsed if isinstance(item, dict) and 'name' in item]
        return []
        
    except (ValueError, SyntaxError, KeyError) as e:
        logger.warning(f"Error parsing features: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in extract_features: {str(e)}")
        return []

def get_director(crew_str):
    """
    Extract director from crew.
    
    Args:
        crew_str: JSON string containing crew information
        
    Returns:
        list: Director name(s)
    """
    try:
        if isinstance(crew_str, str):
            crew = ast.literal_eval(crew_str)
        else:
            crew = crew_str
            
        for member in crew:
            if isinstance(member, dict) and member.get('job') == 'Director':
                return [member['name']]
                
    except (ValueError, SyntaxError, KeyError) as e:
        logger.warning(f"Error parsing crew: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_director: {str(e)}")
    
    return []

def preprocess_data(movies):
    """
    Clean and fuse features into a unified tag-cloud.
    
    Args:
        movies: Raw movies DataFrame
        
    Returns:
        DataFrame: Processed movies with tags
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # Select required columns
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
        missing_cols = [col for col in required_cols if col not in movies.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        movies = movies[required_cols].copy()
        
        # Drop rows with missing values
        initial_count = len(movies)
        movies.dropna(inplace=True)
        logger.info(f"Dropped {initial_count - len(movies)} rows with missing values")
        
        # Standardize 'genres' as clean lists
        logger.info("Extracting features...")
        movies['genres'] = movies['genres'].apply(extract_features)
        movies['keywords'] = movies['keywords'].apply(extract_features)
        movies['cast'] = movies['cast'].apply(lambda x: extract_features(x)[:3])  # Top 3 cast
        movies['crew'] = movies['crew'].apply(get_director)
        
        # Remove spaces for NLP tokenization consistency
        for col in ['genres', 'keywords', 'cast', 'crew']:
            movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Create unified tags
        logger.info("Creating unified tags...")
        movies['tags'] = (
            movies['overview'].fillna('') + ' ' +
            movies['genres'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['cast'].apply(lambda x: ' '.join(x)) + ' ' +
            movies['crew'].apply(lambda x: ' '.join(x))
        )
        movies['tags'] = movies['tags'].apply(lambda x: x.lower())
        
        logger.info(f"Preprocessing complete. Final shape: {movies.shape}")
        return movies[['movie_id', 'title', 'tags', 'genres']]
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        return None

def build_similarity_matrix(movies):
    """
    Vectorize tags and compute cosine similarity.
    
    Args:
        movies: Preprocessed movies DataFrame with tags
        
    Returns:
        tuple: (similarity_matrix, vectorizer, movies_with_coords)
    """
    try:
        logger.info("Building similarity matrix...")
        
        # Vectorize tags
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        logger.info(f"Vectorized to shape: {vectors.shape}")
        
        # Compute Cosine Similarity (optimized to float32 for storage)
        logger.info("Computing cosine similarity...")
        similarity = cosine_similarity(vectors).astype('float32')
        logger.info(f"Similarity matrix shape: {similarity.shape}")
        
        # Precompute 3D Latent Space from tag vectors (Research Grade visualization)
        logger.info("Computing 3D SVD Manifold projection...")
        svd = TruncatedSVD(n_components=3, random_state=42)
        coords = svd.fit_transform(vectors)
        
        movies['dim_x'] = coords[:, 0]
        movies['dim_y'] = coords[:, 1]
        movies['dim_z'] = coords[:, 2]
        
        logger.info("Successfully built similarity matrix and manifold projection")
        return similarity, cv, movies
        
    except Exception as e:
        logger.error(f"Error building similarity matrix: {str(e)}")
        return None, None, None

def save_artifacts(movies, similarity, cv):
    """
    Save all research artifacts.
    
    Args:
        movies: Processed movies DataFrame
        similarity: Similarity matrix
        cv: Fitted CountVectorizer
    """
    try:
        logger.info("Saving artifacts...")
        
        pickle.dump(movies, open('movie_list.pkl', 'wb'))
        logger.info("✅ Saved movie_list.pkl")
        
        pickle.dump(similarity, open('similarity.pkl', 'wb'))
        logger.info("✅ Saved similarity.pkl")
        
        pickle.dump(cv, open('cv.pkl', 'wb'))
        logger.info("✅ Saved cv.pkl")
        
        logger.info("All artifacts saved successfully!")
        
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting CineMatch Research Pipeline")
    logger.info("="*60)
    
    movies_raw = load_tmdb_data()
    
    if movies_raw is None:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)
    
    movies_proc = preprocess_data(movies_raw)
    
    if movies_proc is None:
        logger.error("Failed to preprocess data. Exiting.")
        sys.exit(1)
    
    similarity, cv, movies_final = build_similarity_matrix(movies_proc)
    
    if similarity is None or cv is None or movies_final is None:
        logger.error("Failed to build similarity matrix. Exiting.")
        sys.exit(1)
    
    logger.info(f"Final dataset shape: {movies_final.shape}")
    
    try:
        save_artifacts(movies_final, similarity, cv)
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info("You can now run: streamlit run Movie.py")
    except Exception as e:
        logger.error(f"Failed to save artifacts: {str(e)}")
        sys.exit(1)
