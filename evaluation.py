import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import streamlit as st
import os
import logging

# Setup logging
logger = logging.getLogger('CineMatch.evaluation')

# Deterministic seed for scientific reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def evaluate_recommender(movies_df, method='bow', top_k=10):
    """
    Evaluate recommendation quality using genre overlap as proxy metric.
    
    Args:
        movies_df: DataFrame with 'tags' and 'genres' columns
        method: Vectorization method ('bow', 'tfidf', or 'sbert')
        top_k: Number of recommendations to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        logger.info(f"Evaluating {method.upper()} method...")
        start_time = time.time()
        
        if method == 'bow':
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
            vectors = vectorizer.fit_transform(movies_df['tags'])
        
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            vectors = vectorizer.fit_transform(movies_df['tags'])
        
        elif method == 'sbert':
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                vectors = model.encode(movies_df['tags'].tolist(), show_progress_bar=False)
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        
        similarity_matrix = cosine_similarity(vectors)
        vectorization_time = time.time() - start_time
        
        precisions = []
        # Sample 50 movies for deterministic evaluation
        all_indices = np.arange(len(movies_df))
        sample_indices = np.random.RandomState(RANDOM_SEED).choice(
            all_indices, min(50, len(movies_df)), replace=False
        )
        
        for idx in sample_indices:
            rec_indices = np.argsort(similarity_matrix[idx])[-top_k-1:-1][::-1]
            
            # Ground truth: Standardized genre lists
            query_genres = set(movies_df.iloc[idx]['genres']) 

            relevant_count = 0
            for rec_idx in rec_indices:
                rec_genres = set(movies_df.iloc[rec_idx]['genres'])
                if query_genres & rec_genres:  # If there's any overlap
                    relevant_count += 1
            
            precisions.append(relevant_count / top_k)
        
        avg_latency = (vectorization_time / len(movies_df)) * 1000 
        
        result = {
            'method': method.upper(),
            'precision@k': np.mean(precisions),
            'std_precision': np.std(precisions),
            'latency_ms': avg_latency,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"{method.upper()} - Precision@{top_k}: {result['precision@k']:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error in evaluate_recommender: {str(e)}")
        raise

def run_ablation_study(movies_df):
    """
    Compare BoW vs TF-IDF vs SBERT and log results for research archives.
    
    Args:
        movies_df: DataFrame with movie data
        
    Returns:
        DataFrame: Results comparison
    """
    logger.info("Starting ablation study...")
    results = []
    methods = ['bow', 'tfidf']
    
    for method in methods:
        try:
            metrics = evaluate_recommender(movies_df, method=method, top_k=10)
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating {method}: {str(e)}")
            st.error(f"Failed to evaluate {method.upper()}: {str(e)}")
            
    # Try SBERT (may fail if not installed or too slow)
    try:
        logger.info("Attempting SBERT evaluation (this may take a while)...")
        metrics = evaluate_recommender(movies_df, method='sbert', top_k=10)
        results.append(metrics)
        logger.info("SBERT evaluation completed successfully")
    except ImportError:
        logger.warning("SBERT not available (sentence-transformers not installed)")
        st.warning("⚠️ SBERT evaluation skipped: sentence-transformers not installed")
    except Exception as e:
        logger.error(f"SBERT evaluation failed: {str(e)}")
        st.warning(f"⚠️ SBERT evaluation failed: {str(e)}")
    
    if not results:
        logger.error("No evaluation results generated")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Persistent Logging for Research Verification
    try:
        results_df.to_csv('evaluation_results.csv', index=False)
        logger.info("Results saved to evaluation_results.csv")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
    
    return results_df

if __name__ == "__main__":
    import pickle
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 Loading artifacts for evaluation...")
    try:
        movies = pickle.load(open('movie_list.pkl', 'rb'))
        print(f"✅ Loaded {len(movies)} movies")
        
        print("🚀 Starting Ablation Study...")
        results = run_ablation_study(movies)
        
        if not results.empty:
            print("\n--- RESEARCH RESULTS ---")
            print(results[['method', 'precision@k', 'latency_ms']])
            print("\n✅ Results saved to evaluation_results.csv")
        else:
            print("❌ No results generated")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ movie_list.pkl not found. Run preprocess.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)
